import numpy as np
import linsolve
import aipy
import catdata
import linsolve
import h5py
import time
from collections import OrderedDict

pol2ind = {'xx':0, 'yy':1}

def _mk_key(pixel, srcid, timeid):
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

def unravel_pix(dim, coord):
    """
    Returns the unraveled/flattened pixel of any (m, n) position
    or coordinates on any square 2D-grid

    Parameters
    ----------
    dim : int
        Dimension of the square 2D grid.

    coord : tuple of int
        Coordinates (m, n) for which to calculate the  flattened index.

    Returns
    -------
    index of the flattened array corresponding to the coordinates (m, n).
    """

    return (coord[0] * dim) + coord[1]

def rotate_mat(theta):
    """
    Rotate coordinates or pixels by theta degrees

    Parameters
    ----------
    theta : float
            Angle by which the coordinates or pixels will be rotated.
    """
    
    return np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def get_weights(azalts, bm_pix, theta=0, flip=1):
    """
    Returns the four closest pixels to the azimuth-altitude values on the 2D
    grid.

    Parameters
    ----------
    azalts : ndarray
        2D array consisting of the azimuth and alitutes values in degrees.
    """

    # selecting the four closest pixels
    tx, ty, tz = aipy.coord.azalt2top([azalts[0, :] * np.pi/180., azalts[1, :] * np.pi/180.])
    tx, ty = np.dot(rotate_mat(theta), np.array([tx, ty]))
    tx = flip * tx 
    tx_px = tx * 0.5 * bm_pix + 0.5 * bm_pix
    ty_px = ty * 0.5 * bm_pix + 0.5 * bm_pix
    tx_px0 = np.floor(tx_px).astype(np.int)
    tx_px1 = np.clip(tx_px0 + 1, 0, bm_pix -1)
    ty_px0 = np.floor(ty_px).astype(np.int)
    ty_px1 = np.clip(ty_px0 + 1, 0, bm_pix -1)

    x0y0 = np.array([tx_px0, ty_px0], dtype=np.int)
    x0y1 = np.array([tx_px0, ty_px1], dtype=np.int)
    x1y0 = np.array([tx_px1, ty_px0], dtype=np.int)
    x1y1 = np.array([tx_px1, ty_px1], dtype=np.int)

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

def eval_linear_sol(sol, bm_pix):
    """
    Evalutes the solutions to output the beam values into a
    2D grid.

    sol : dict
        Dictionary containing the solutions, returned by the solver.
    
    bm_pix : int
        Pixel number of the output 2D beam.
    """

    obs_beam = np.zeros((bm_pix**2), dtype=float)
    for key in sol.keys():
        px = int(key.strip('b'))
        obs_beam[px] = sol.get(key)

    obs_beam.shape = (bm_pix, bm_pix)
    return obs_beam

def eval_nonlinear_sol(sol, nsrcs, bm_pix):
    """
    Evalutes the solutions to output the beam values into a
    2D grid and the model flux values.

    sol : dict
        Dictionary containing the solutions, returned by the solver.
    """

    obs_beam = np.zeros((bm_pix**2), dtype=float)
    fluxvals = np.zeros((2, nsrcs))

    k = 0
    for key in sol[1].keys():
        if key[0] == 'b':
            px = int(key.strip('b'))
            obs_beam[px] = sol[1].get(key)
        if key[0] == 'I':
            fluxvals[0, k] = key[1::]
            fluxvals[1, k] = sol[1].get(key)
            k += 1

    obs_beam.shape = (bm_pix, bm_pix)
    return fluxvals, obs_beam

def get_A(ls):
    """
    Returns the A matrix used to solve the system of linear equations

            y = A.x.
    """

    return ls.get_A()

def svd(A):
    """
    Decomposes m x n matrix through single value decomposition

    Parameters
    ----------
    A : numpy array/ matrix of floats.
    """

    A.shape = (A.shape[0], A.shape[1])
    AtA = np.dot(A.T.conj(), A)
    # decomposes A matrix
    U, S, V = np.linalg.svd(AtA)

    return U, S, V

def remove_degen(ls, beam, bm_pix, threshold=3e-4):
    """
    Remove degeneracies using single value decomposition. It removes all eigenvalue modes
    above the specified threshold.

    sol : dict
        Dictionary containing the solutions, returned by the solver.

    threshold : float
        Threshold value after which all the eigenvalue modes will be discarded.
    """

    A = get_A(ls)
    U, S, V = svd(A)

    # determining the cutoff threshold for bad eigen modes
    total = sum(S)
    var_exp = np.array([(i / total) * 100 for i in sorted(S, reverse=True)])
    # selecting the cutoff eigen-mode
    cutoff_mode = np.where(var_exp < threshold)[0][0]
    print ('Removing all eigen modes above {}'.format(cutoff_mode))

    for i in xrange(cutoff_mode, len(S)):
        emode = np.array([U[ls.prm_order['b%d' % px], i] if ls.prm_order.has_key('b%d'%px) else 0 for px in xrange(bm_pix**2)])
        emode.shape = (bm_pix, bm_pix)
        beam -= np.sum(beam * emode) * emode.conj()

    return beam

class BeamSolveBase(object):
    def __init__(self, cat=None, bm_pix=60):
        """
        Object to store a catalogue or multiple astronomical sources to solve for the beam parameters

        Parameters
        ----------
        cat : catData object
            catData object containing data (flux densities) of sources along with the necessary meta data.
    
        bm_pix : int
            Pixel number of the output 2D beam.
        """

        self.cat = cat
        self.bm_pix = bm_pix
        self.consts = OrderedDict()
        self.eqs = OrderedDict()
        self.ls = None

    def construct_linear_sys(self, mflux=[], theta=[0], flip=[1]):
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
        mflux : list or np.ndarray
            List or array containing the model flux values to be used as I_mod.
        """

        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        obs_vals = self.cat.data_array[0]

        for i in xrange(nsrcs):
            for th in theta:
                for fl in flip:      
                    ps, ws = get_weights(self.cat.azalt_array[:, i, :], self.bm_pix, th)
                    for j in xrange(nfits):
                        I_s = obs_vals[i, j]
                        if np.isnan(I_s): continue
                        c = {_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                        eq = ' + '.join([_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                            + '*b%d'%(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in xrange(4)])
                        self.eqs[eq] = I_s / mflux[i]
                        self.consts.update(c)        

    def construct_nonlinear_sys(self, mflux, bvals, theta=[0], flip=[1], constrain=False):
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

        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        obs_vals = self.cat.data_array[0]

        bvals_f = bvals.flatten()
        self.sol_dict = OrderedDict()

        for i in xrange(nsrcs):   
            self.sol_dict['I%d'%i] = mflux[i]
            for th in theta:
                for fl in flip:
                    ps, ws = get_weights(self.cat.azalt_array[:, i, :], self.bm_pix, th, fl)
                    for j in xrange(nfits):
                        I_s = obs_vals[i, j]
                        if np.isnan(I_s): continue
                        c = {_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                        eq = ' + '.join([_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                            + '*b%d'% (unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))) + '*I%d'%i for p in xrange(4)])
                        self.eqs[eq] = I_s
                        self.consts.update(c)
                        for p in xrange(4):
                            bpix = unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))
                            self.sol_dict['b%d'%bpix] = bvals_f[bpix]

        # constraining the center pixel to be one
        if constrain:
            self.eqs['100*b%d'%(unravel_pix(self.bm_pix, (int(self.bm_pix/2.), int(self.bm_pix/2.))))] = 100.0

    def solve(self, solver=None, conv_crit=1e-10, maxiter=50):
        """
        Solves for system of linear equations using Linear or system of non-linear
        eqautions using LinProduct solver. 
        
        Parameters
        ----------
        solver : str
            Name of solver to use to solve for the system of linear or non-linear equations.
            The solver can be either 'Linear' (linear) or 'LinProduct' (non-linear).

        conv_crit : float
            Convergence of chi^2. The solver will stop iterating if conv_crit is
            reached. Defaulr is 1e-10.

        maxiter : int
            Number of iterations the solver will go through before reaching the 
            solutions. Default is 50.
        """
        
        time0 = time.time()
        
        if solver == 'Linear':
            self.ls = linsolve.LinearSolver(self.eqs, **self.consts)
            sol = self.ls.solve(verbose=True)

        if solver == 'LinProduct':
            self.ls = linsolve.LinProductSolver(self.eqs, sol0=self.sol_dict, **self.consts)
            sol = self.ls.solve_iteratively(conv_crit, maxiter, verbose=True)

        print ('Time Elapsed: {:0.2f} seconds'.format(time.time() - time0))
        return sol


class BeamSolveCross(object):
    def __init__(self, cat=None, bm_pix=60):
        """
        Object to store a catalogue or multiple astronomical sources to solve for the beam parameters

        Parameters
        ----------
        cat : catData object
            catData object containing data (flux densities) of sources along with the necessary meta data.

        bm_pix : int
            Pixel number of the output 2D beam.
        """

        self.cat = cat
        self.bm_pix = bm_pix
        self.consts = OrderedDict()
        self.eqs = OrderedDict()
        self.ls = None

    def construct_linear_sys(self, mflux_xx=[], mflux_yy=[], theta_xx=[0], theta_yy=[np.pi/2], flip=[1]):
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
        mflux : list or np.ndarray
            List or array containing the model flux values to be used as I_mod.
        """

        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        obsvals_xx = self.cat.data_array[0]
        obsvals_yy = self.cat.data_array[1]

        for i in xrange(nsrcs):
            for th in theta_xx:
                for fl in flip:
                    ps, ws = get_weights(self.cat.azalt_array[:, i, :], self.bm_pix, th, fl)
                    for j in xrange(nfits):
                        I_s = obsvals_xx[i, j]
                        if np.isnan(I_s): continue
                        c = {_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                        eq = ' + '.join([_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                            + '*b%d'%(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in xrange(4)])
                        self.eqs[eq] = I_s / mflux_xx[i]
                        self.consts.update(c)

            for th in theta_yy:
                for fl in flip:
                    ps, ws = get_weights(self.cat.azalt_array[:, i, :], self.bm_pix, th, fl)
                    for j in xrange(nfits):
                        I_s = obsvals_yy[i, j]
                        if np.isnan(I_s): continue
                        c = {_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                        eq = ' + '.join([_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                            + '*b%d'%(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in xrange(4)])
                        self.eqs[eq] = I_s / mflux_yy[i]
                        self.consts.update(c)

    def construct_nonlinear_sys(self, mflux_xx=[], mflux_yy=[], bvals=[], theta_xx=[0], theta_yy=[np.pi/2], flip=[1], constrain=False):
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

        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        obsvals_xx = self.cat.data_array[0]
        obsvals_yy = self.cat.data_array[1]

        bvals_f = bvals.flatten()
        self.sol_dict = OrderedDict()

        for i in xrange(nsrcs):
            self.sol_dict['I%d'%i] = 0.5 * (mflux_xx[i] + mflux_yy[i])
            for th in theta_xx:
                for fl in flip:
                    ps, ws = get_weights(self.cat.azalt_array[:, i, :], self.bm_pix, th, fl)
                    for j in xrange(nfits):
                        I_s = obsvals_xx[i, j]
                        if np.isnan(I_s): continue
                        c = {_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                        eq = ' + '.join([_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                            + '*b%d'% (unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))) + '*I%d'%i for p in xrange(4)])
                        self.eqs[eq] = I_s
                        self.consts.update(c)
                        for p in xrange(4):
                            bpix = unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))
                            self.sol_dict['b%d'%bpix] = bvals_f[bpix]

            for th in theta_yy:
                for fl in flip:
                    ps, ws = get_weights(self.cat.azalt_array[:, i, :], self.bm_pix, th, fl)
                    for j in xrange(nfits):
                        I_s = obsvals_yy[i, j]
                        if np.isnan(I_s): continue
                        c = {_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                        eq = ' + '.join([_mk_key(unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                            + '*b%d'% (unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))) + '*I%d'%i for p in xrange(4)])
                        self.eqs[eq] = I_s
                        self.consts.update(c)
                        for p in xrange(4):
                            bpix = unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j]))
                            self.sol_dict['b%d'%bpix] = bvals_f[bpix]

        # constraining the center pixel to be one
        if constrain:
            self.eqs['100*b%d'%(unravel_pix(self.bm_pix, (int(self.bm_pix/2.), int(self.bm_pix/2.))))] = 100.0

    def solve(self, solver=None, conv_crit=1e-10, maxiter=50):
        """
        Solves for system of linear equations using Linear or system of non-linear
        eqautions using LinProduct solver.

        Parameters
        ----------
        solver : str
            Name of solver to use to solve for the system of linear or non-linear equations.
            The solver can be either 'Linear' (linear) or 'LinProduct' (non-linear).

        conv_crit : float
            Convergence of chi^2. The solver will stop iterating if conv_crit is
            reached. Defaulr is 1e-10.

        maxiter : int
            Number of iterations the solver will go through before reaching the
            solutions. Default is 50.
        """

        time0 = time.time()

        if solver == 'Linear':
            self.ls = linsolve.LinearSolver(self.eqs, **self.consts)
            sol = self.ls.solve(verbose=True)

        if solver == 'LinProduct':
            self.ls = linsolve.LinProductSolver(self.eqs, sol0=self.sol_dict, **self.consts)
            sol = self.ls.solve_iteratively(conv_crit, maxiter, verbose=True)

        print ('Time Elapsed: {:0.2f} seconds'.format(time.time() - time0))
        return sol
