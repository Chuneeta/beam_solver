import numpy as np
import linsolve
import aipy
import catdata
import linsolve
import h5py
import time
from collections import OrderedDict

pol2ind = {'xx':0, 'yy':1}

class BeamSolving(object):
    def __init__(self, cat=None, bm_pix=60):
        """
        Object to store a catalogue or multiple astronomical sources to solve for the beam parameters
        
        Parameters
        ----------
        cat : catData object 
            catData object containing data (flux densities) of sources along with the necessary meta data
        """
 
        self.cat = cat
        self.bm_pix = bm_pix
        self.consts = OrderedDict()
        self.eqs = OrderedDict()
        self.ls = None

    def _mk_key(self, px, sid, tid):
        """
        Generates key to represent the beam pixel

        Parameters
        ----------
        px : int

        sid : int

        tid : int
             
        """

        return 'w%d_s%d_t%d' % (px, sid, tid)

    def _unravel_pix(self, npix, pos):
        """
        Returns the unraveled/flattened pixel of any (m, n) position 
        or coordinates on a 2D-grid

        Parameters
        ----------
        npix : int

        coord : tuple of int
        """

        i, j = pos[0], pos[1]
        return (i * npix) + j

    def rotation_mat(self, theta=0):
        """
        Return rotation matrix used to rotate the beam pixels by theta

        Parameters
        ----------
        theta : float
            Angle in radians by which the coordinates/pixels will be rotated.
            Default is 0. 
        """

        return np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            
    def get_weights(self, azalts, theta=0):
        """
        """
        
        # selecting the four closest pixels
        tx, ty, tz = aipy.coord.azalt2top([azalts[0, :] * np.pi/180., azalts[1, :] * np.pi/180.])
        #tx_r, ty_r, tz_r = np.array(self.rotation_mat(theta), np.array(tx, ty)) # rotating the coordinates by theta
        tx_px = tx * 0.5 * self.bm_pix + 0.5 * self.bm_pix
        ty_px = ty * 0.5 * self.bm_pix + 0.5 * self.bm_pix
        tx_px0 = np.floor(tx_px).astype(np.int)
        tx_px1 = np.clip(tx_px0 + 1, 0, self.bm_pix -1)
        ty_px0 = np.floor(ty_px).astype(np.int)
        ty_px1 = np.clip(ty_px0 +1, 0, self.bm_pix -1)

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

    def _soldict(self, fluxvals=[]):
        """
        """

        soldict = OrderedDict()
        assert len (fluxvals) == len(cat.ras), 'Length of flux value guesses should be the same as the number of sources'
        for i in xrange(cat.ras.shape[0]):
            self.soldict['I%d'%i] = fluxvals[i]
            ps, ws = self.get_weights(self.azalt_array[:, i, :])
            for j in xrange(self.cat.nfits):
            	for p in xrange(4):
                    sol_dict['b%d'%(self.unravel_pix(self.bm_pix, ps[p][0, j], ps[p][1, j]))] = 0.0 
        
        return sol_dict         

    def construct_linear_sys(self, flux_type='peak', pol='xx'):
        """
        """

        if flux_type == 'peak':
            obs_vals = self.cat.pflux_array[pol2ind[pol]]
            model_vals = self.cat.pcorr_array[pol2ind[pol]]
        elif flux_type == 'total':
            obs_vals = self.cat.tflux_array[pol2ind[pol]]
            model_vals = self.cat.tcorr_array[pol2ind[pol]]
        else:
            raise ValueError('Flux type is not recognized')

        nfits = self.cat.Nfits

        for i in xrange(self.cat.ras.shape[0]):
            ps, ws = self.get_weights(self.cat.azalt_array[:, i, :])
            for j in xrange(nfits):
                I_s = obs_vals[i, j]
                if np.isnan(I_s): continue
                c = {self._mk_key(self._unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] for p in xrange(4)}
                eq = ' + '.join([self._mk_key(self._unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                     + '*b%d'%(self._unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in xrange(4)])

                self.eqs[eq] = I_s / model_vals[i]
                self.consts.update(c)

    def check_rotation(self, thetas):
        """
        """

        if len(thetas) == 1:
            raise ValueError('theta shoud be of length > 1.')
        
        diff = np.unique(np.diff(thetas))
        if diff.size > 1:
            raise ValueError('the rotation should be the same for all thetas')

        if diff[0] != 2 * np.pi:
            raise ValueError('only 180 (2 pi) rotation are allowed for a single polarization')
        
    def construct_nonlinear_sys(self, flux_type='peak', pol=['xx'], xxrot=[0], yyrot=[], dual_pol='False'):
        """
        """
        if flux_type == 'peak':
            obs_vals = self.cat.pflux_array
        elif flux_type == 'total':
            obs_vals = self.cat.tflux_array
        else:
            raise ValueError('Flux type is not recognized')

        if pol.size == 1:
            obs_vals = obs_vals[pol2ind[pol]]
        nfits = self.cat.Nfits
        for i in xrange(self.cat.ras.shape[0]):
            ps, ws = self.get_weights(self.cat.azalt_array[:, i, :])
            for j in xrange(nfits):
                I_s = obs_vals[i, j]
                if np.isnan(I_s): continue
                c = {self._mk_key(self._unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j): ws[p][j] / I_s for p in xrange(4)}
                eq = ' + '.join([self._mk_key(self._unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])), i, j)
                        + '*b%d'%(self._unravel_pix(self.bm_pix, (ps[p][0, j], ps[p][1, j])))  for p in xrange(4)])

                self.eqs[eq] = 1
                self.consts.update(c)          
           
    def solve(self, solver=None, fluxvals=[], conv_crit=1e-10, maxiter=50):
        """
        """
        time0 = time.time()

        if solver == 'LinProduct':
            sol0 = self. _soldict(fluxvals=fluxvals)
            self.ls = linsolve.LinProductSolver(self.eqs, sol0=sol, **self.consts)
            sol = self.ls.solve_iteratively(maxiter=maxiter, conv_crit=conv_crit, verbose=True)

        elif solver == 'Linear':
            self.ls = linsolve.LinearSolver(self.eqs, **self.consts)
            sol = self.ls.solve(verbose=True)
    
        else:
            raise ValueError('solver is not recognized.')

        print ('Time Elapsed: {:0.2f} seconds'.format(time.time() - time0))
        return sol

    def eval_sol(self, sol):
        """
        """
        obs_beam = np.zeros((self.bm_pix**2), dtype=float)
        for key in sol.keys():
            px = int(key.strip('b'))
            obs_beam[px] = sol.get(key)

        return obs_beam

    def get_A(self):
        return self.ls.get_A()
 
    def svd(self, A):
        """
        Decomposes m x n matrix through single value decomposition

        Parameters
        ----------
        A : numpy array/ matrix of floats
        """

        A.shape = (A.shape[0], A.shape[1])
        AtA = np.dot(A.T.conj(), A)
        # decomposes A matrix 
        U, S, V = np.linalg.svd(AtA)    

        return U, S, V

    def remove_degeneracies(self, sol, threshold=0.003):
        """
        """
        #evaluating the solutions
        obs_beam = self.eval_sol(sol)
        obs_beam.shape = (self.bm_pix, self.bm_pix)

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
