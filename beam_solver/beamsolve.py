import numpy as np
import linsolve
import aipy
import catdata
import linsolve
import h5py
import time
from collections import OrderedDict

pol2ind = {'xx':0, 'yy':1}

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

        #if not isinstance(cat, catdata.catData):
        #    raise ValueError('catalog cat needs to a catData object.')

        if not isinstance(bm_pix, (int, np.int)):
            try:
                bm_pix = int(bm_pix)
            except:
                raise ValueError('bm_pix needs to be an integer value.')

        self.cat = cat
        self.bm_pix = bm_pix
        self.consts = OrderedDict()
        self.eqs = OrderedDict()
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
        """
    
        if not isinstance(pixel, (int, np.int)):
            try:
                pixel = int(pixel)
            except:
                raise ValueError('pixel should be an integer.')

        if not isinstance(srcid, (int, np.int)):
            try:
                srcid = int(srcid)
            except:
                raise ValueError('srcid should be an integer.')
        
        if not isinstance(timeid, (int, np.int)):
            try:
                timeid = int(timeid)
            except: 
                raise ValueError('timeid should be an integer.') 

        return 'w%d_s%d_t%d' % (pixel, srcid, timeid)

    def unravel_pix(self, dim, coord):
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
    
        if not isinstance(dim, (int, np.int)):
            try:
                dim = np.int(dim)
            except:
                raise ValueError('dim should be an interger value.')
        
        if not isinstance(coord, tuple):
            try:
                coord = tuple(coord)
            except:
                raise ValueError('coord should be a tuple of integers.')

        if not isinstance(coord[0], (int, np.int)):
            try:
                i = int(coord[0])
            except:
                raise ValueError('instances of coord should be integer value.') 
        else:
            i = coord[0]

        if not isinstance(coord[1], (int, np.int)):
            try:
                j = int(coord[1])
            except:
                raise ValueError('instances of coord should be integer value.')
        else:
            j = coord[1]
          
        return (i * dim) + j

    def get_weights(self, azalts):
        """
        Returns the four closest pixels to the azimuth-altitude values on the 2D
        grid.

        Parameters
        ----------
        azalts : ndarray
            2D array consisting of the azimuth and alitutes values in degrees.
        """
        
        # checks if the azalts is of 2D shape
        if len(azalts) != 2:
            raise ValueError('azalts should be a 2-dimensional array.')

        # selecting the four closest pixels
        tx, ty, tz = aipy.coord.azalt2top([azalts[0, :] * np.pi/180., azalts[1, :] * np.pi/180.])
        tx_px = tx * 0.5 * self.bm_pix + 0.5 * self.bm_pix
        ty_px = ty * 0.5 * self.bm_pix + 0.5 * self.bm_pix
        tx_px0 = np.floor(tx_px).astype(np.int)
        tx_px1 = np.clip(tx_px0 + 1, 0, self.bm_pix -1)
        ty_px0 = np.floor(ty_px).astype(np.int)
        ty_px1 = np.clip(ty_px0 + 1, 0, self.bm_pix -1)

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

    def construct_linear_sys(self, flux_type='peak'):
        """
        Construct a linear system of equations of the form
    
                    I_mod * A  = I_obs.
        
        where I_mod is model flux value, A is primary beam value and I_obs is our measurement.
        We decompose A such that

                A = a1 * w1 + a2 * w2 + a3 * w3 + a4 * w4

        where (a1, a2, a3, a4) are the four closest pixel values of the beam to the azimuth-altitude
        value of the source at a given time and (w1, w2, w3, w4) are the corrsponding weights.

        Parameters
        ---------
        flux_type : str
            Flux type can be either peak or total (integrated flux value).
            Default is peak.
        """

        nfits = self.cat.Nfits
        nsrcs = self.cat.Nsrcs

        if flux_type == 'peak':
            obs_vals = self.cat.pflux_array[0]
            model_vals = self.cat.pcorr_array[0]
        elif flux_type == 'total':
            obs_vals = self.cat.tflux_array[0]
            model_vals = self.cat.tcorr_array[0]
        else:
            raise ValueError('flux type is not recognized should be either "peak" or "total".')

        # checks shape of model values
        assert model_vals.shape == (nsrcs,), 'tcorr_array or pcorr_array should be of shape {}.'.format(model_vals.shape) 
       
        if np.array_equal(np.unique(model_vals[0]), np.array([0])):
            raise ValueError('tcorr_array/pcorr_array should be non-zero values.')

        for i in xrange(self.cat.ras.shape[0]):
            ps, ws = self.get_weights(self.cat.azalt_array[:, i, :])
            for j in xrange(nfits):
                I_s = obs_vals[i, j]
                I_s = np.nan_to_num(I_s)
                c = {self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                eq = ' + '.join([self._mk_key(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                     + '*b%d'%(self.unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in xrange(4)])

                self.eqs[eq] = I_s / model_vals[i]
                self.consts.update(c)

    def solve(self, solver=None, conv_crit=1e-10, maxiter=50):
        """
        Solves for system of linear equations using Linear or system of non-linear
        eqautions using LinProduct solver. 
        
        Parameters
        ----------
        solver : str
            Name of solver to use to solve for the system of linear equations.

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

        else:
            raise ValueError('solver is not recognized.')

        print ('Time Elapsed: {:0.2f} seconds'.format(time.time() - time0))
        return sol

    def eval_sol(self, sol):
        """
        Evalutes the solutions to output the beam values into a
        2D grid.

        sol : dict
            Dictionary containing the solutions, returned by the solver.
        """
        
        if not isinstance(sol, dict):
            raise ValueError('sol should be a dictionary.')

        obs_beam = np.zeros((self.bm_pix**2), dtype=float)
        for key in sol.keys():
            px = int(key.strip('b'))
            obs_beam[px] = sol.get(key)
        
        obs_beam.shape = (self.bm_pix, self.bm_pix)

        return obs_beam

    def get_A(self):
        """
        Returns the A matrix used to solve the system of linear equations
        
                 y = A.x.
        """

        return self.ls.get_A()

    def svd(self, A):
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

    def remove_degen(self, sol, threshold=3e-4):
        """
        Remove degeneracies using single value decomposition. It removes all eigenvalue modes
        above the specified threshold.

        sol : dict
            Dictionary containing the solutions, returned by the solver.

        threshold : float
            Threshold value after which all the eigenvalue modes will be discarded.
        """

        if not isinstance(threshold, (int, np.int, float, np.float)):
            try:
                threshold = float(threshold)
            except:
                raise ValueError('Threshold should be either a float or integer value.')

        #evaluating the solutions
        obs_beam = self.eval_sol(sol)

        A = self.get_A()
        U, S, V = self.svd(A)

        # determining the cutoff threshold for bad eigen modes
        total = sum(S)
        var_exp = np.array([(i / total) * 100 for i in sorted(S, reverse=True)])
        # selecting the cutoff eigen-mode
        cutoff_mode = np.where(var_exp < threshold)[0][0]
        print ('Removing all eigen modes above {}'.format(cutoff_mode))

        for i in xrange(cutoff_mode, len(S)):
            emode = np.array([U[self.ls.prm_order['b%d' % px], i] if self.ls.prm_order.has_key('b%d'%px) else 0 for px in xrange(self.bm_pix**2)])
            emode.shape = (self.bm_pix, self.bm_pix)
            obs_beam -= np.sum(obs_beam * emode) * emode.conj()

        return obs_beam
