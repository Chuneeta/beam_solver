import numpy as np
import linsolve
import aipy
import time
from collections import OrderedDict
from scipy.sparse.linalg import inv
from scipy.sparse import csc_matrix

class BeamOnly():
    def __init__(self, cat=None, bm_pix=61):
        """
        Object that stores the flux catalog containing the flux values for one
        polarization and solves for the primary beam only.
        """
        self.cat = cat
        self.bm_pix = bm_pix
        self.eqs = OrderedDict()
        self.consts = OrderedDict()
        self.sol_dict = OrderedDict()
        self.sigma = OrderedDict()
        self.ls = None

    def _mk_key(self, pixel, srcid, timeid):
        """
        Generates key to represent the beam pixel which include the source id,
        timestamp and pixel.
        Parameters
        ----------
        pixel : int
            Pixel from the 2D grid beam.
        srcid : int
            Source identity.
        timeid : int
            Time identity or timestamps.
        Returns
        -------
            String corresponding the given parameters.
        """
        return 'w%d_s%d_t%d' % (pixel, srcid, timeid)

    def unravel_pix(self, ndim, coord):
        """
        Returns the unraveled/flattened pixel of any (m, n) position
        or coordinates on any square 2D-grid

        Parameters
        ----------
        coord : tuple of int
            Coordinates (m, n) for which to calculate the  flattened index.

        Returns
        -------
        index of the flattened array corresponding to the coordinates (m, n).
        """
        return (coord[0] * ndim) + coord[1]

    def rotate_mat(self, theta):
        """
        Rotate coordinates or pixels by theta degrees

        Parameters
        ----------
        theta : float
            Angle by which the coordinates or pixels will be rotated.
        """
        return np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def get_weights(self, azalts, theta, flip):
        """
        Returns the four closest pixels to the azimuth-altitude values on the 2D
        grid.
        Parameters
        ----------
        azalts : ndarray
            2D array consisting of the azimuth and alitutes values in radians.
        """
        # selecting the four closest pixels
        tx, ty, tz = aipy.coord.azalt2top([azalts[0, :], azalts[1, :]])
        tx_r, ty_r = np.dot(self.rotate_mat(theta), np.array([tx, ty]))
        tx_r *= flip
        tx_px = tx_r * 0.5 * self.bm_pix + 0.5 * self.bm_pix - 0.5
        ty_px = ty_r * 0.5 * self.bm_pix + 0.5 * self.bm_pix - 0.5
        tx_px = np.where(tx_px < 0, 0, tx_px)
        ty_px = np.where(ty_px < 0, 0, ty_px)
        tx_px0 = np.floor(tx_px).astype(np.int)
        tx_px1 = np.clip(tx_px0 + 1, 0, self.bm_pix - 1)
        ty_px0 = np.floor(ty_px).astype(np.int)
        ty_px1 = np.clip(ty_px0 + 1, 0, self.bm_pix - 1)        
        x0y0 = np.array([tx_px0, ty_px0])
        x0y1 = np.array([tx_px0, ty_px1])
        x1y0 = np.array([tx_px1, ty_px0])
        x1y1 = np.array([tx_px1, ty_px1])
        # defining the weights
        fx = tx_px - tx_px0
        fy = ty_px - ty_px0
        w0 = (1 - fx) * (1 - fy)
        w1 = fx * (1 - fy)
        w2 = (1 - fx) * fy
        w3 = fx * fy       
        ps = [x0y0, x0y1, x1y0, x1y1]
        ws = [w0, w1, w2, w3] 
        return ps, ws

    def _mk_eq(self, ps, ws, obs_flux, catalog_flux, sigma, srcid, timeid, equal_wgts, **kwargs):
        """
        Constructs equations that will form the linear system of equations.
        Parameters
        ---------
        ps : nd array
            Numpy array containing the four closest pixel numbers corresponding to the
            alt-az position of the source
        ws : ns array
            Numpy array contining the weights corresponding to the pixel numbers
        obs_flux : float
            Measured or observed flux value
        catalog_flux : float
            Catalog or corrected flux value obtained using the flux values from the catalog.
            Refer to catdata.calc_catalog_flux.
        srcid : int
            Source identity.
        timeid : int
            Time identity or timestamps.
        """
        if equal_wgts is True:
            flux_wgts = 1
            divisor = catalog_flux
        else:
            flux_wgts = catalog_flux
            divisor = 1
        i = srcid; j = timeid
        c = {self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0,j], ps[p][1,j])), i, j): flux_wgts * ws[p][j] for p in range(4)}
        eq = ' + '.join([self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
            + '*b%d'%(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in range(4)])
        if eq not in self.eqs:
            self.eqs[eq] = obs_flux / divisor
            self.consts.update(c)
            self.sigma[eq] = sigma

    def calc_catalog_flux(self, beam_model, pol):
        """
        Returns catalog flux
        """
        return self.cat.calc_catalog_flux(beam_model, pol)

    def _build_solver(self, noise=None, **kwargs):
        """
        Builds linsolve solver
        """
        if not noise is None:
            print  ('Yupiii')
            N = self.get_noise_matrix(noise_type=kwargs['noise_type'])
            self.ls = linsolve.LinearSolverNoise(self.eqs, N, **self.consts)
        else:
            
            self.ls = linsolve.LinearSolver(self.eqs, **self.consts)

    def get_A(self, ls):
        """
        Returns the A matrix used to solve the system of linear equations

            y = A.x.

        Parameters
        ----------
        ls : linsolve instance
            instance of linsolve solver containing the linear system of equations.
        """
        return ls.get_A()

    def svd(self, ls, A):
        """
        Decomposes m x n matrix through single value decomposition

        Parameters
        ----------
		ls : linsolve instance
            instance of linsolve solver containing the linear system of equations.
        A : numpy array/ matrix of floats.
        """
        A.shape = (A.shape[0], A.shape[1])
        AtA = np.dot(A.T.conj(), A)
        # decomposes A matrix
        U, S, V = np.linalg.svd(AtA)

        return U, S, V

    def remove_degen(self, ls, obsbeam, threshold=5e-6, key='b'):
        """
        Remove degeneracies using single value decomposition. It removes all eigenvalue modes
        above the specified threshold.
		ls : instance of linsolve
			instance of linsolve solver containing the linear system of equations.
        obsbeam : 2D numpy array
            2-dimensional array containing the beam values.
        threshold : float
            Threshold value after which all the eigenvalue modes will be discarded.
        key : str
            Keyword or naming convention used to denote the beam pixels in the solver object.
             Default is 'b'.
        """
        A = self.get_A(ls)
        U, S, V = self.svd(ls, A)

        # determining the cutoff threshold for bad eigen modes
        total = sum(S)
        var_exp = np.array([(i / total) for i in sorted(S, reverse=True)])
        # selecting the cutoff eigen-mode
        cutoff_mode = np.where(var_exp < threshold)[0][0]
        print ('Removing all eigen modes above {}'.format(cutoff_mode))

        for i in range(cutoff_mode, len(S)):
            emode = np.array([U[ls.prm_order['%s%d' % (key, px)], i] if '%s%d'% (key, px) in ls.prm_order.keys() else 0 for px in range(self.bm_pix**2)])
            emode.shape = (self.bm_pix, self.bm_pix)
            obsbeam -= np.sum(obsbeam * emode) * emode.conj()

        return obsbeam

    def add_eqs(self, catalog_flux, theta=[0], flip=[1], polnum=0, flux_thresh=0, equal_wgts=True, **kwargs):
        """
        Construct a linear system of equations of the form

            I_mod * A  = I_obs.

        where I_mod is model flux value, A is primary beam value and I_obs is our measurement.
        We decompose A such that
 
            A = a1 * w1 + a2 * w2 + a3 * w3 + a4 * w4

        where (a1, a2, a3, a4) are the four closest pixel values of the beam to the azimuth-altitude
        value of the source at a given time and (w1, w2, w3, w4) are the corrsponding weights.
        Parameters
        ----------
        catalog_flux : list or np.ndarray
            List or array containing the model/catalog flux values to be used as I_mod.
 
    	theta : float
       	"""
        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        obs_vals = self.cat.data_array[polnum]
        err_vals = self.cat.error_array[polnum]
        for i in range(nsrcs):
            for th in theta:
                for fl in flip:
                    ps, ws = self.get_weights(self.cat.azalt_array[:, i, :], th, fl)
                    for j in range(nfits):
                        I_s = obs_vals[i, j]
                        if np.isnan(I_s) or I_s < flux_thresh:continue
                        self._mk_eq(ps, ws, I_s, catalog_flux[i], err_vals[i,j], i, j, equal_wgts, **kwargs)

    def solve(self, **kwargs):
        """
        Solves for the linear system of equations
        """
        self._build_solver(**kwargs)
        sol = self.ls.solve(verbose=True)
        return sol        

    def eval_sol(self, sol):
        """
        Evaluates the solutions to output the beam values into a
        2D grid.

        Parameters
        ----------
        sol : dict
            Dictionary containing the solutions, returned by the solver.
        """
        obs_beam = np.zeros((self.bm_pix**2), dtype=float)
        for key in sol.keys():
            if key[0] == 'b':
                px = int(key.strip('b'))
                obs_beam[px] = sol.get(key)

        obs_beam.shape = (self.bm_pix, self.bm_pix)
        return obs_beam

    def _uncorr_noise_matrix(self):
        """
        Returns uncorrelated noise matrix, i.e matrix containing only the variances 
        associated with the flux measurements
        """
        noise_n = len(self.sigma)
        noise_array = np.zeros((noise_n, noise_n))
        np.fill_diagonal(noise_array, [self.sigma[key] for key in self.sigma.keys()])
        return noise_array.T

    def _partial_corr_noise_matrix(self):
        """
        Returns partially correlated noise matrix, i.e matrix containing only the variances
        associated with the flux measurements and covariances of adjacent flux measurements.
        """
        noise_n = len(self.sigma)
        noise_array = np.zeros((noise_n, noise_n))
        keys = list(self.sigma.keys())
        # filling diagonals
        np.fill_diagonal(noise_array, [self.sigma[key] for key in keys])
        # filling off-diagonal terms
        np.fill_diagonal(noise_array[1:], [0.5 * self.sigma[keys[i]] for i in range(len(keys))])
        np.fill_diagonal(noise_array[:, 1:], [0.5 * self.sigma[keys[1:][i]] for i in range(len(keys) - 1)]) 
        return noise_array.T

    def _full_corr_noise_matrix(self):
        """
        Returns fully correlated noise matrix, i.e matrix containing only the variances
        and covariances associated with the flux measurements.
        """
        noise_n = len(self.sigma)
        noise_array = np.zeros((noise_n, noise_n))
        row0 = [self.sigma[key] for key in self.sigma.keys()]
        noise_array = np.repeat(row0, noise_n)
        noise_array = noise_array.reshape((noise_n, noise_n))
        return noise_array.T

    def get_noise_matrix(self, noise_type):
        """
        Return the noise matrix containing error associated with the flux measurements
    
        Parameters
        ----------
        noise_type: string
            Type of noise or error, can be 'uncorr', 'partial' or 'uncorr'.
            'uncorr' assumes the the error measurements associated with the measurements
            are uncorrelated with each other.
            'partial' assumes that the error measurements associated with the mesurements
            are correlated with adjacent measurements.
            'corr' considers correlation between all the flux measurements.
        """
        if noise_type == 'corr': noise_matrix = self._full_corr_noise_matrix()
        elif noise_type == 'partial': noise_matrix = self._partial_corr_noise_matrix()
        else: noise_matrix = self._uncorr_noise_matrix()
        return noise_matrix

    def _eval_error(self, ls, noise_type):
        """
        Evaluates the error (AtN^-1A)^-1

        Parameters
        ----------
        ls: instance of linsolve
            instance of linsolve solver containing the linear system of equations.
                noise_type: string
            Type of noise or error, can be 'uncorr', 'partial' or 'uncorr'.
            'uncorr' assumes the the error measurements associated with the measurements
            are uncorrelated with each other.
            'partial' assumes that the error measurements associated with the mesurements
            are correlated with adjacent measurements.
            'corr' considers correlation between all the flux measurements.
            Default is 'uncorr'.
        """
        A = self.get_A(ls)
        An = (A - np.min(A))/(np.max(A) - np.min(A))
        An_sparse = csc_matrix(A[:, :, 0])
        N = self.get_noise_matrix(noise_type=noise_type)
        N_sparse = csc_matrix(N)
        Ninv = inv(N_sparse)
        At = An_sparse.T
        AtNi = At.dot(Ninv)
        AtNiA = AtNi.dot(An_sparse)
        AtNiAi = inv(AtNiA).todense()
        return AtNiAi

    def eval_beam_error(self, sol, ls, noise_type='uncorr'):
        """
        Evaluates the error associated with the beam solutions

        Parameters
        ----------
        sol: dict
            dictionary return by linsolve containing the beam solutions
        ls: instance of linsolve
            instance of linsolve solver containing the linear system of equations.
                noise_type: string
            Type of noise or error, can be 'uncorr', 'partial' or 'uncorr'.
            'uncorr' assumes the the error measurements associated with the measurements
            are uncorrelated with each other.
            'partial' assumes that the error measurements associated with the mesurements
            are correlated with adjacent measurements.
            'corr' considers correlation between all the flux measurements.
            Default is 'uncorr'.
        """
        AtNiAi = self._eval_error(ls, noise_type)
        beam_error = np.diag(np.linalg.inv(AtNiAi))
        beam_error_mat = np.zeros((self.bm_pix**2), dtype=float)
        for ii, key in enumerate(list(sol.keys())):
            if key[0] == 'b':
                px = int(key.strip('b'))
                beam_error_mat[px] = beam_error[ii]
        beam_error_mat.shape = (self.bm_pix, self.bm_pix)
        return beam_error_mat

class BeamCat(BeamOnly):
    def __init__(self, cat=None, bm_pix=61):
        """
        Object that stores the flux catalog containing the flux values for one
        polarization and solves for both the true flux values of the sources and
        the primary beam.
        """
        BeamOnly.__init__(self, cat, bm_pix)

    def _mk_eq(self, ps, ws, obs_flux, catalog_flux, sigma, srcid, timeid, equal_wgts, **kwargs):
        """
        Constructs equations that will form the linear system of equations.
        Parameters
        ----------
        ps : numpy.nd array
            Numpy array containing the four closest pixel numbers corresponding to the
            alt-az position of the source
        ws : ns array
            Numpy array continuing the weights corresponding to the pixel numbers
        obs_flux : float
            Measured or observed flux value
        catalog_flux : float
            Catalog or corrected flux value obtained using the flux values from the catalog.
            Refer to catdata.calc_catalog_flux.
        srcid : int
            Source identity.
        timeid : int
            Time identity or timestamps.
        """
        i = srcid ; j = timeid
        bvals = kwargs['bvals'].flatten()
        self.sol_dict['I%d'%i] = catalog_flux
        c = {self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0,j], ps[p][1,j])), i, j): ws[p][j]
                    for p in range(4)}
        if equal_wgts:
            eq1 = ' + '.join([self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
            + '*b%d'% (self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))) for p in range(4)])
            eq = 'I%d * (%s)'%(i, eq1)
        else:
            eq = ' + '.join([self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                + '*b%d'% (self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))) + '*I%d'%i for p in range(4)])
        if eq not in self.eqs:
            self.eqs[eq] = obs_flux
            self.sigma[eq] = sigma
            self.consts.update(c)
        for p in range(4):
            bpix = int(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))
            self.sol_dict['b%d'%bpix] = bvals[bpix]

    def add_eqs(self, catalog_flux, theta=[0], flip=[1], polnum=0, flux_thresh=0, equal_wgts=True, **kwargs):
        """
        Construct a non linear system of equations of the form

            I_mod * A  = I_obs.

        where I_mod is model flux value, A is primary beam value and I_obs is our measurement.
        We decompose A such that

            A = a1 * w1 + a2 * w2 + a3 * w3 + a4 * w4

        where (a1, a2, a3, a4) are the four closest pixel values of the beam to the azimuth-altitude
        value of the source at a given time and (w1, w2, w3, w4) are the corrsponding weights.

        In the non-linear case, we solve for both I_mod and A, however initial guesses are required
        to start the iteration.
        Parameters
        ----------
        catalog_flux : list or np.ndarray
            List or array containing the model/catalog flux values to be used as I_mod.
        """
        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        obs_vals = self.cat.data_array[polnum]
        err_vals = self.cat.error_array[polnum] 
        for i in range(nsrcs):
            for th in theta:
                for fl in flip:
                    ps, ws = self.get_weights(self.cat.azalt_array[:, i, :], th, fl)
                    for j in range(nfits):
                        I_s = obs_vals[i, j]
                        if np.isnan(I_s) or I_s < flux_thresh: continue
                        self._mk_eq(ps, ws, I_s, catalog_flux[i], err_vals[i,j], i, j, equal_wgts, **kwargs)
                        
    def add_constrain(self, srcid, val):
        """
        Add constrain to the flux value in the system of linear equations
        """
        self.eqs['I%d'%srcid] = val

    def _build_solver(self, norm_weight, **kwargs):
        """
        Builds linsolve solver

		Parameters
		----------
		norm_weight: float
			Value specified to force the desired constrained. Default is 100.
        """
        constrain = kwargs.pop('constrain', False)
        # constraining the center pixel to be one
        if constrain:	
            self.eqs['%d*b%d'%(norm_weight, self.unravel_pix(self.bm_pix, (int(self.bm_pix/2.), int(self.bm_pix/2.))))] = norm_weight
        self.ls = linsolve.LinProductSolver(self.eqs, sol0=self.sol_dict, **self.consts)

    def solve(self, maxiter=50, conv_crit=1e-11, norm_weight=100, **kwargs):
        """
        Solves for the primary beam
        """
        self._build_solver(norm_weight, **kwargs)
        meta, sol = self.ls.solve_iteratively(maxiter=maxiter, conv_crit=conv_crit, verbose=True)
        return meta, sol

    def eval_sol(self, sol):
        """
        Evaluates the solutions to output the beam values into a
        2D grid and the model flux values.

        Parameters
        ----------
        sol : dict
            Dictionary containing the solutions, returned by the solver.
        """
        obs_beam = BeamOnly(cat=self.cat, bm_pix=self.bm_pix).eval_sol(sol[1])
        fluxvals = np.zeros((2, self.cat.Nsrcs))
        for key in sol[1].keys():
            if key[0] == 'I':
                ind = int(key[1::])
                fluxvals[0, ind] = ind
                fluxvals[1, ind] = sol[1].get(key)
        return fluxvals, obs_beam

    def eval_error(self, sol, ls, constrain=False):
        A = self.get_A(ls)
        if constrain:
            A = A[:-1] # removing the constrained equation
        # evaluating (AtN^-1A)^-1
        A_n = (A - np.min(A))/(np.max(A) - np.min(A))
        inv_noise = np.linalg.inv(self.get_noise_matrix())
        AtNA = np.dot(np.dot(A_n[:, :, 0].T.conj(), inv_noise), A_n[:, :, 0])
        errors = np.diag(np.linalg.inv(AtNA))
        flux_error = np.zeros((self.cat.Nsrcs))
        beam_error_mat = np.zeros((self.bm_pix**2), dtype=float)
        for ii, key in enumerate(sol.keys()):
            if key[0] == 'I':
                ind = int(key[1::])
                flux_error[ind] = errors[ii]
            else:
                px = int(key.strip('b'))
                beam_error_mat[px] = errors[ii]

        beam_error_mat.shape = (self.bm_pix, self.bm_pix)
        return flux_error, beam_error_mat

class BeamOnlyCross(BeamOnly):
    def __init__(self, cat=None, bm_pix=61):
        """
        Object that stores the flux catalog containing the flux values for xx and yy
        polarization and solves for the primary beam only using both polarizations.
        """
        BeamOnly.__init__(self, cat, bm_pix)
        
    def add_eqs(self, catalog_flux_xx, catalog_flux_yy, theta_xx=[0], theta_yy=[np.pi/2], flip_xx=[1], flip_yy=[1], flux_thresh=0, equal_wgts=True, **kwargs):
        """
        Construct a linear system of equations of the form

            I_mod * A  = I_obs.

        where I_mod is model flux value, A is primary beam value and I_obs is our measurement.
        We decompose A such that

            A = a1 * w1 + a2 * w2 + a3 * w3 + a4 * w4

        where (a1, a2, a3, a4) are the four closest pixel values of the beam to the azimuth-altitude
        value of the source at a given time and (w1, w2, w3, w4) are the corrsponding weights.

        Parameters
        ----------
        catalog_flux : list or np.ndarray
            List or array containing the model/catalog flux values to be used as I_mod.
        """
        BeamOnly.add_eqs(self, catalog_flux=catalog_flux_xx, theta=theta_xx, flip=flip_xx, flux_thresh=flux_thresh, polnum=0, equal_wgts=equal_wgts)
        BeamOnly.add_eqs(self, catalog_flux=catalog_flux_yy, theta=theta_yy, flip=flip_yy, flux_thresh=flux_thresh, polnum=1, equal_wgts=equal_wgts)

    def solve(self, **kwargs):
        """
        Solves for the linear system of equations
        """
        self._build_solver(**kwargs)
        sol = self.ls.solve(verbose=True)
        return sol
        
class BeamCatCross(BeamCat):
    def __init__(self, cat=None, bm_pix=61):
        """
        Object that stores the flux catalog containing the flux values for xx and yy
        polarization and solves for both the flux values of the sources and the primary 
        beam using both polarizations.
        """
        BeamCat.__init__(self, cat, bm_pix)

    def add_eqs(self, catalog_flux_xx, catalog_flux_yy, theta_xx=[0], theta_yy=[np.pi/2], flip_xx=[1], flip_yy=[1], polnum=0, equal_wgts=True, **kwargs):
        """
        Construct a non linear system of equations of the form

            I_mod * A  = I_obs.

        where I_mod is model flux value, A is primary beam value and I_obs is our measurement.
        We decompose A such that

            A = a1 * w1 + a2 * w2 + a3 * w3 + a4 * w4

        where (a1, a2, a3, a4) are the four closest pixel values of the beam to the azimuth-altitude
        value of the source at a given time and (w1, w2, w3, w4) are the corrsponding weights.

        In the non-linear case, we solve for both I_mod and A, however initial guesses are required
        to start the iteration.

        Parameters
        ----------
        mflux : list or np.ndarray
            List or array containing the model flux values to be used as initial guesses for I_mod.
            Default is list of ones.

        bvals : 2D numpy array
            Array containing initial guesses for the beam. Default is zeros.

        constrain : boolean
            If True, constrained the center pixel to be one. It will error out if the sources are not
            transiting zenith. Default is False.
        """
        BeamCat.add_eqs(self, catalog_flux=catalog_flux_xx, theta=theta_xx, flip=flip_xx, polnum=0, equal_wgts=equal_wgts, **kwargs)
        BeamCat.add_eqs(self, catalog_flux=catalog_flux_yy, theta=theta_yy, flip=flip_yy, polnum=1, equal_wgts=equal_wgts, **kwargs)

    def solve(self, maxiter=50, conv_crit=1e-11, norm_weight=100, **kwargs):
        self._build_solver(norm_weight = norm_weight, **kwargs)
        sol = self.ls.solve_iteratively(maxiter=maxiter, conv_crit=conv_crit, verbose=True)
        return sol
