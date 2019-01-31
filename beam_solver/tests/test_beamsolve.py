from beam_solver.data import DATA_PATH
from collections import OrderedDict
from beam_solver import catdata as cd
from beam_solver import beamsolve as bs
from beam_solver import beam_utils as bt
import nose.tools as nt
import numpy as np
import collections
import aipy
import os
import copy
import pylab
import linsolve

def mk_key(px, i, t):
    return 'w%d_s%d_t%d' % (px, i, t)

def unravel_pix(n, i,j):
    return (i*n) + j

def recenter(a, c):
    """Slide the (0,0) point of matrix a to a new location tuple c.  This is
    useful for making an image centered on your screen after performing an
    inverse fft of uv data."""
    s = a.shape
    c = (c[0] % s[0], c[1] % s[1])
    if np.ma.isMA(a):
        a1 = np.ma.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = np.ma.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    else:
        a1 = np.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = np.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    return a2

def get_LM(dim, center=(0,0), res=1):
        """Get the (l,m) image coordinates for an inverted UV matrix."""
        M,L = np.indices((dim, dim))
        L,M = np.where(L > dim/2, dim-L, -L), np.where(M > dim/2, M-dim, M)
        L,M = L.astype(np.float32)/dim/res, M.astype(np.float32)/dim/res
        mask = np.where(L**2 + M**2 >= 1, 1, 0)
        L,M = np.ma.array(L, mask=mask), np.ma.array(M, mask=mask)
        return recenter(L, center), recenter(M, center)

def get_top(dim, center=(0,0), res=1):
    """Return the topocentric coordinates of each pixel in the image."""
    x,y = get_LM(dim, center, res)
    z = np.sqrt(1 - x**2 - y**2)
    return x,y,z

def rotate(theta, point):
        bm_pix = 31
        x0 = int(0.5 * bm_pix)
        y0 = x0
        x, y = point[0], point[1]
        xr = np.cos(theta) * (x - x0) - np.sin(theta) * (y - y0) + x0
        yr = np.sin(theta) * (x - x0) + np.cos(theta) * (y - y0) + y0
        return np.array([xr, yr])

def rotate_mat(theta):
    """
    Rotate coordinates or pixels by theta degrees

    Parameters
    ----------
    theta : float
        Angle by which the coordinates or pixels will be rotated.
    """
    return np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# beamfile
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

# xx fitsfiles
fitsfile1_xx = os.path.join(DATA_PATH, '2458115.23736.xx.fits')
fitsfile2_xx = os.path.join(DATA_PATH, '2458115.24482.xx.fits')
fitsfiles_xx = [fitsfile1_xx, fitsfile2_xx]

# yy fitsfiles
fitsfile1_yy = os.path.join(DATA_PATH, '2458115.23736.yy.fits')
fitsfile2_yy = os.path.join(DATA_PATH, '2458115.24482.yy.fits')
fitsfiles_yy = [fitsfile1_yy, fitsfile2_yy]

# right ascension and declination values
ras = [30.01713089, 27.72922349, 36.75248962, 34.2415497, 78.3776346, 74.03785837]
decs = [-30.88211818, -29.53377208, -30.63958257, -29.93990039, -30.48595805, -30.08651873]

# generating beam
beam_xx = bt.get_fitsbeam(beamfits, 151e6, 'xx')
beam_yy = bt.get_fitsbeam(beamfits, 151e6, 'yy')

# generating catalog
catd = cd.catData()
catd.gen_catalog(fitsfiles_xx, ras, decs)

class Test_BeamFunc():
    def test_init(self):
        bms = bs.BeamOnly(catd)
        nt.assert_equal(bms.bm_pix, 60)
        nt.assert_equal(bms.cat, catd)
        bms = bs.BeamOnly(catd, 30)
        nt.assert_equal(bms.bm_pix, 30)
        nt.assert_equal(bms.cat, catd)

    def test_mk_key(self):
        bms = bs.BeamOnly(catd)
        key = bms._mk_key(0, 0, 0)
        nt.assert_equal(key, 'w0_s0_t0')
        key = bms._mk_key(0, 1, 4)
        nt.assert_equal(key, 'w0_s1_t4')

    def test_unravel_pix(self):
        bms = bs.BeamOnly(catd)
        ind = bms.unravel_pix(60, (0, 0))
        nt.assert_equal(ind, 0)
        ind = bms.unravel_pix(60, (0, 5))
        nt.assert_equal(ind, 5)    
        bms = bs.BeamOnly(catd, bm_pix=30)
        ind = bms.unravel_pix(30, (0, 5))
        nt.assert_equal(ind, 5)
        ind = bms.unravel_pix(30, (2, 5))
        nt.assert_equal(ind, 65)

    def test_rotate_mat(self):
        bms = bs.BeamOnly(catd)
        theta = 0
        mat0 = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        mat = bms.rotate_mat(theta)
        np.testing.assert_almost_equal(mat, mat0)
        theta = np.pi
        mat0 = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        mat = bms.rotate_mat(theta)
        np.testing.assert_almost_equal(mat, mat0)
        theta = np.pi/4
        mat0 = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        mat = bms.rotate_mat(theta)
        np.testing.assert_almost_equal(mat, mat0)
    	theta = 2 * np.pi
        mat0 = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        mat = np.array([[1., 0.], [0., 1.]])
        np.testing.assert_almost_equal(mat, mat0)

    def test_rotate_coord(self):
        bms = bs.BeamOnly(catd, bm_pix=31)
        rcoord = bms.rotate_coord(0, np.array([0, 0]))
        np.testing.assert_almost_equal(rcoord, np.array([0, 0]))
        rcoord = bms.rotate_coord(np.pi/2, np.array([15., 15.]))
        np.testing.assert_almost_equal(rcoord, np.array([15., 15.]))
        rcoord = bms.rotate_coord(-1 * np.pi/2, np.array([15., 15.]))
        np.testing.assert_almost_equal(rcoord, np.array([15., 15.]))
        rcoord = bms.rotate_coord(np.pi/2, np.array([14., 15.]))
        np.testing.assert_almost_equal(rcoord, np.array([15., 14.]))
        rcoord = bms.rotate_coord(-1 * np.pi/2, np.array([14., 15.]))
        np.testing.assert_almost_equal(rcoord, np.array([15., 16.]))
        rcoord = bms.rotate_coord(2 * 2 * np.pi, np.array([14., 15.]))
        np.testing.assert_almost_equal(rcoord, np.array([14., 15.]))

    def test_get_weights(self):
        bms = bs.BeamOnly(catd, 4)
        ps, ws = bms.get_weights(np.array([[0], [0]]), 0, 1)
        bms = bs.BeamOnly(catd, 12)
        ps, ws = bms.get_weights(np.array([[0], [0]]), 0, 1)
        nt.assert_equal(ps[0][0][0], ps[1][0][0])
        nt.assert_equal(ps[2][0][0], ps[3][0][0])
        nt.assert_equal(ps[0][1][0], ps[2][1][0])
        nt.assert_equal(ps[1][1][0], ps[3][1][0])
        nt.assert_almost_equal(np.sum(ws), 1)
        bms = bs.BeamOnly(catd)
        ps, ws = bms.get_weights(np.array([[0], [0]]), 0, 1)
        nt.assert_equal(ps[0][0][0], ps[1][0][0])
        nt.assert_equal(ps[2][0][0], ps[3][0][0])
        nt.assert_equal(ps[0][1][0], ps[2][1][0])
        nt.assert_equal(ps[1][1][0], ps[3][1][0])
        nt.assert_almost_equal(np.sum(ws), 1)

class Test_BeamOnly():
    def test_solve(self):
        consts = OrderedDict()
        eqs = OrderedDict()
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        bm_pix = 31
        bm_true = np.zeros((bm_pix, bm_pix), dtype=float)
        bmx, bmy = np.indices(bm_true.shape)
        mu = 15.; sigma=3.
        bm_true = np.exp(-((bmx-mu)**2 + (bmy-mu)**2)/ (2 * sigma**2))
        tx0, ty0, tz0 = get_top(bm_pix, center=(15, 15), res=0.5)
        tx00, ty00, tz00 = tx0.flatten(), ty0.flatten(), tz0.flatten()
        indices = np.arange(bm_pix**2)
        azs, alts = aipy.coord.top2azalt([np.array(tx0[:, :]), np.array(ty0[:, :]), np.array(tz0[:, :])])
        #azs, alts = aipy.coord.top2azalt([np.array(tx0[15:16,15:16]), np.array(ty0[15:16, 15:16]), np.array(tz0[15:16, 15:16])])
        azs = np.rad2deg(azs); alts = np.rad2deg(alts)
        _sh = azs.shape
        fluxvals = np.random.random(_sh[0]) + 10
        newdata = np.zeros((2, _sh[0], _sh[1]))
        for i in range(_sh[0]):
            tx, ty, tz = aipy.coord.azalt2top([np.deg2rad(azs[i, :]), np.deg2rad(alts[i, :])])
            tx_px = tx * 0.5 * bm_pix + 0.5 * bm_pix
            tx_px0 = np.floor(tx_px).astype(np.int)
            tx_px1 = np.clip(tx_px0 + 1, 0, bm_pix - 1)
            fx = tx_px - tx_px0
            ty_px = ty * 0.5 * bm_pix + 0.5 * bm_pix
            ty_px0 = np.floor(ty_px).astype(np.int)
            ty_px1 = np.clip(ty_px0 + 1, 0, bm_pix - 1)
            fy = ty_px - ty_px0
            x0y0 = np.array([tx_px0, ty_px0], dtype=np.int)
            x0y1 = np.array([tx_px0, ty_px1], dtype=np.int)
            x1y0 = np.array([tx_px1, ty_px0], dtype=np.int)
            x1y1 = np.array([tx_px1, ty_px1], dtype=np.int)

            w0 = (1 - fx) * (1 - fy)
            w1 = fx * (1 - fy)
            w2 = (1 - fx) * fy
            w3 = fx * fy

            ps = [x0y0, x0y1, x1y0, x1y1]
            ws = [w0, w1, w2, w3]
            
            for j in range(_sh[1]):
                A_s = (bm_true[tx_px0[j],ty_px0[j]] * w0[j] + bm_true[tx_px0[j],ty_px1[j]] * w1[j] + bm_true[tx_px1[j],ty_px0[j]] * w2[j] + bm_true[tx_px1[j], ty_px1[j]] * w3[j])
                I_s = fluxvals[i] * A_s
                newdata[0, i, j] = I_s
                c = {mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j): ws[p][j] for p in xrange(len(ps))}
                eq = ' + '.join([mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j) + \
                    '*b%d'%(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j])) for p in xrange(len(ps))])
                eqs[eq] = I_s / fluxvals[i]
                consts.update(c)

        eqs_noise = {k:v for k,v in eqs.items()}
        ls = linsolve.LinearSolver(eqs_noise, **consts)
        sol0 = ls.solve(verbose=True)
        interp2d = np.zeros((bm_pix**2))
        for key in sol0.keys():
            px = int(key.strip('b'))
            interp2d[px] = sol0.get(key)

        # testing beamonly solver
        catd.azalt_array = np.zeros((2, _sh[0], _sh[1]))
        catd.azalt_array[0, :, :] = azs
        catd.azalt_array[1, :, :] = alts
        catd.Nsrcs = _sh[0]
        catd.Nfits = _sh[1]
        catd.data_array = newdata
        np.testing.assert_almost_equal(catd.data_array, newdata)
        np.testing.assert_almost_equal(catd.azalt_array[0, :, :], azs)
        np.testing.assert_almost_equal(catd.azalt_array[1, :, :], alts)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxvals)
        nt.assert_equal(eqs, bms.eqs)
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)
        diff = obsbeam - bm_true
        np.testing.assert_allclose(diff, np.zeros((bm_pix, bm_pix)), rtol=1e-05, atol=1e-05)
        np.testing.assert_allclose(obsbeam, bm_true, rtol=1e-05, atol=1e-05)
        cleanbeam = bms.remove_degen(bms.ls, obsbeam)
        np.testing.assert_allclose(cleanbeam, bm_true, rtol=1e-05, atol=1e-05)

class Test_BeamCat():
	def test_solve(self):
		consts = OrderedDict()
		eqs = OrderedDict()
		sol_dict = OrderedDict()
		catd = cd.catData()
		catd.gen_catalog(fitsfiles_xx, ras, decs)
		bm_pix = 31
		interp2d = np.zeros((bm_pix**2))
		bm_true = np.zeros((bm_pix, bm_pix), dtype=float)
		bmx, bmy = np.indices(bm_true.shape)
		mu = 15.; sigma=3.
		bm_true = np.exp(-((bmx-mu)**2 + (bmy-mu)**2)/ (2 * sigma**2))
		tx0, ty0, tz0 = get_top(bm_pix, center=(15, 15), res=0.5)
		tx00, ty00, tz00 = tx0.flatten(), ty0.flatten(), tz0.flatten()
		indices = np.arange(bm_pix**2)
		#azs, alts = aipy.coord.top2azalt([np.array(tx0[:, :]), np.array(ty0[:, :]), np.array(tz0[:, :])])
		azs, alts = aipy.coord.top2azalt([np.array(tx0[15:16,15:16]), np.array(ty0[15:16, 15:16]), np.array(tz0[15:16, 15:16])])
		azs = np.rad2deg(azs); alts = np.rad2deg(alts)
		_sh = azs.shape
		fluxvals = np.random.random(_sh[0]) + 10
		newdata = np.zeros((1, _sh[0], _sh[1]))

		for i in range(_sh[0]):
			tx, ty, tz = aipy.coord.azalt2top([np.deg2rad(azs[i,:]), np.deg2rad(alts[i,:])])
			tx_px = tx * 0.5*bm_pix + 0.5*bm_pix
			tx_px0 = np.floor(tx_px).astype(np.int)
			tx_px1 = np.clip(tx_px0+1,0,bm_pix-1)#tx_px0 + 1
			fx = tx_px - tx_px0
			ty_px = ty * 0.5*bm_pix + 0.5*bm_pix
			ty_px0 = np.floor(ty_px).astype(np.int)
			ty_px1 = np.clip(ty_px0 + 1,0,bm_pix-1)#ty_px0 + 1
			fy = ty_px - ty_px0
			x0y0 = np.array([tx_px0, ty_px0], dtype=np.int)
			x0y1 = np.array([tx_px0, ty_px1], dtype=np.int)
			x1y0 = np.array([tx_px1, ty_px0], dtype=np.int)
			x1y1 = np.array([tx_px1, ty_px1], dtype=np.int)

			w0 = (1 - fx) * (1 - fy)
			w1 = fx * (1 - fy)
			w2 = (1 - fx) * fy
			w3 = fx * fy

			ps = [x0y0, x0y1, x1y0, x1y1]
			ws = [w0, w1, w2, w3]

			sol_dict['I%d'%i] = fluxvals[i]
			for j in range(_sh[1]):
				weights = w0[j] + w1[j] + w2[j] + w3[j]
				A_s = (bm_true[tx_px0[j],ty_px0[j]] * w0[j] + bm_true[tx_px0[j],ty_px1[j]] * w1[j] \
				  + bm_true[tx_px1[j],ty_px0[j]] * w2[j] + bm_true[tx_px1[j], ty_px1[j]] * w3[j])
				I_s = fluxvals[i] * A_s
				newdata[0, i, j] = I_s
				c = {mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j): ws[p][j] for p in xrange(len(ps))}
				eq = ' + '.join([mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j) + \
				 '*b%d'%(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j])) + '*I%d'%i for p in xrange(len(ps))])
				eqs[eq] = I_s
				consts.update(c)

				for p in xrange(len(ps)):
					sol_dict['b%d'%(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]))] = 1.0

		eqs_noise = {k:v for k,v in eqs.items()}
		ls = linsolve.LinProductSolver(eqs_noise, sol0=sol_dict, **consts)
		sol0 = ls.solve_iteratively(verbose=True)

		for key in sol0[1].keys():
			if key[0] == 'b':
		   		px = int(key.strip('b'))
		   		interp2d[px] = sol0[1].get(key)

		interp2d = interp2d.reshape((bm_pix, bm_pix))
		catd.azalt_array = np.zeros((2, _sh[0], _sh[1]))
		catd.azalt_array[0, :, :] = azs
		catd.azalt_array[1, :, :] = alts
		catd.Nsrcs = _sh[0]
		catd.Nfits = _sh[1]
		catd.data_array = newdata
		np.testing.assert_almost_equal(catd.data_array, newdata)
		np.testing.assert_almost_equal(catd.azalt_array[0, :, :], azs)
		np.testing.assert_almost_equal(catd.azalt_array[1, :, :], alts)
		bmc = bs.BeamCat(cat=catd, bm_pix=31)
		bmc.add_eqs(catalog_flux=fluxvals, bvals=np.ones((bm_pix, bm_pix)), constrain=True)
		sol = bmc.solve()
		fluxvals, obsbeam = bmc.eval_sol(sol)
		np.testing.assert_allclose(obsbeam, interp2d)
		cleanbeam = bmc.remove_degen(bmc.ls.ls, obsbeam)
""
class Test_BeamOnlyCross():
    def test_solve(self):
        consts = OrderedDict()
        eqs = OrderedDict()
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        bm_pix = 31
        bm_true = np.zeros((bm_pix, bm_pix), dtype=float)
        bmx, bmy = np.indices(bm_true.shape)
        mu = 15.; sigma=3.
        bm_true = np.exp(-((bmx-mu)**2 + (bmy-mu)**2)/ (2 * sigma**2))
        tx0, ty0, tz0 = get_top(bm_pix, center=(15, 15), res=0.5)
        tx00, ty00, tz00 = tx0.flatten(), ty0.flatten(), tz0.flatten()
        indices = np.arange(bm_pix**2)
        azs, alts = aipy.coord.top2azalt([np.array(tx0[:, :]), np.array(ty0[:, :]), np.array(tz0[:, :])])
        #azs, alts = aipy.coord.top2azalt([np.array(tx0[15:16,15:16]), np.array(ty0[15:16, 15:16]), np.array(tz0[15:16, 15:16])])
        azs = np.rad2deg(azs); alts = np.rad2deg(alts)
        _sh = azs.shape
        fluxvals = np.random.random(_sh[0]) + 10
        newdata = np.zeros((2, _sh[0], _sh[1]))	
        for i in range(_sh[0]):
            tx, ty, tz = aipy.coord.azalt2top([np.deg2rad(azs[i, :]), np.deg2rad(alts[i, :])])
            tx_px = tx * 0.5 * bm_pix + 0.5 * bm_pix
            tx_px0 = np.floor(tx_px).astype(np.int)
            tx_px1 = np.clip(tx_px0 + 1, 0, bm_pix - 1)
            fx = tx_px - tx_px0
            ty_px = ty * 0.5 * bm_pix + 0.5 * bm_pix
            ty_px0 = np.floor(ty_px).astype(np.int)
            ty_px1 = np.clip(ty_px0 + 1, 0, bm_pix - 1)
            fy = ty_px - ty_px0
            x0y0 = np.array([tx_px0, ty_px0], dtype=np.int)
            x0y1 = np.array([tx_px0, ty_px1], dtype=np.int)
            x1y0 = np.array([tx_px1, ty_px0], dtype=np.int)
            x1y1 = np.array([tx_px1, ty_px1], dtype=np.int)

            w0 = (1 - fx) * (1 - fy)
            w1 = fx * (1 - fy)
            w2 = (1 - fx) * fy
            w3 = fx * fy

            ps = [x0y0, x0y1, x1y0, x1y1]
            ws = [w0, w1, w2, w3]

        
            flip = -1
            theta = np.pi / 2
            rx0y0 = np.dot(rotate_mat(theta), np.array([tx_px0, ty_px0], dtype=np.int))
            rx0y1 = np.dot(rotate_mat(theta), np.array([tx_px0, ty_px1], dtype=np.int))
            rx1y0 = np.dot(rotate_mat(theta), np.array([tx_px1, ty_px0], dtype=np.int))
            rx1y1 = np.dot(rotate_mat(theta), np.array([tx_px1, ty_px1], dtype=np.int))

            #rx0y0 = rotate(theta, np.array([x0y0[0,:], x0y0[1,:]])).astype(int)
            #rx0y1 = rotate(theta, np.array([x0y1[0,:], x0y1[1,:]])).astype(int)
            #rx1y0 = rotate(theta, np.array([x1y0[0,:], x1y0[1,:]])).astype(int)
            #rx1y1 = rotate(theta, np.array([x1y1[0,:], x1y1[1,:]])).astype(int)

            rx0y0[0] = flip * rx0y0[0]
            rx0y1[0] = flip * rx0y1[0]
            rx1y0[0] = flip * rx1y0[0]
            rx1y1[0] = flip * rx1y1[0]

            rtx_px0 = rx0y0[0, :].astype(int)
            rty_px0 = rx0y0[1, :].astype(int)
            rtx_px1 = rx1y1[0, :].astype(int)
            rty_px1 = rx1y1[1, :].astype(int)

            rps = [rx0y0, rx0y1, rx1y0, rx1y1]

            for j in range(_sh[1]):
                A_s = (bm_true[tx_px0[j],ty_px0[j]] * w0[j] + bm_true[tx_px0[j],ty_px1[j]] * w1[j] + bm_true[tx_px1[j],ty_px0[j]] * w2[j] + bm_true[tx_px1[j], ty_px1[j]] * w3[j])
                I_s = fluxvals[i] * A_s
                newdata[0, i, j] = I_s
                c = {mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j): ws[p][j] for p in xrange(len(ps))}
                eq = ' + '.join([mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j) + \
                    '*b%d'%(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j])) for p in xrange(len(ps))])
                eqs[eq] = I_s / fluxvals[i]
                consts.update(c)
                
                A_s = (bm_true[rtx_px0[j],rty_px0[j]] * w0[j] + bm_true[rtx_px0[j],rty_px1[j]] * w1[j] + bm_true[rtx_px1[j],rty_px0[j]] * w2[j] + bm_true[rtx_px1[j], rty_px1[j]] * w3[j])
                I_s = fluxvals[i] * A_s
                newdata[1, i, j] = I_s
                c = {mk_key(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j]), i, j): ws[p][j] for p in xrange(len(ps))}
                eq = ' + '.join([mk_key(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j]), i, j) + \
                    '*b%d'%(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j])) for p in xrange(len(ps))])
                eqs[eq] = I_s / fluxvals[i]
                consts.update(c)

        eqs_noise = {k:v for k,v in eqs.items()}
        ls = linsolve.LinearSolver(eqs_noise, **consts)
        sol0 = ls.solve(verbose=True)
        interp2d = np.zeros((bm_pix**2))
        for key in sol0.keys():
            px = int(key.strip('b'))
            interp2d[px] = sol0.get(key)

        # testing beamonlycross solver
        catd.azalt_array = np.zeros((2, _sh[0], _sh[1]))
        catd.azalt_array[0, :, :] = azs
        catd.azalt_array[1, :, :] = alts
        catd.Nsrcs = _sh[0]
        catd.Nfits = _sh[1]
        catd.data_array = newdata
        np.testing.assert_almost_equal(catd.data_array, newdata)
        np.testing.assert_almost_equal(catd.azalt_array[0, :, :], azs)
        np.testing.assert_almost_equal(catd.azalt_array[1, :, :], alts)
        bms = bs.BeamOnlyCross(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux_xx=fluxvals, catalog_flux_yy=fluxvals, theta_yy=[np.pi/2], flip_yy=[-1])
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)
        diff = obsbeam - interp2d.reshape((bm_pix, bm_pix))
        np.testing.assert_allclose(diff, np.zeros((bm_pix, bm_pix)), rtol=1e-04, atol=1e-04)
        np.testing.assert_allclose(obsbeam, interp2d.reshape((bm_pix, bm_pix)), rtol=1e-04, atol=1e-04)
        cleanbeam = bms.remove_degen(bms.ls, obsbeam)
        #np.testing.assert_allclose(cleanbeam, bm_true, rtol=1e-03, atol=1e-03)

"""
class Test_BeamCatCross():
    def test_solve(self):
        consts = OrderedDict()
        eqs = OrderedDict()
        sol_dict = OrderedDict()
        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        bm_pix = 31
        interp2d = np.zeros((bm_pix**2))
        bm_true = np.zeros((bm_pix, bm_pix), dtype=float)
        bmx, bmy = np.indices(bm_true.shape)
        mu = 15.; sigma=3.
        bm_true = np.exp(-((bmx-mu)**2 + (bmy-mu)**2)/ (2 * sigma**2))
        tx0, ty0, tz0 = get_top(bm_pix, center=(15, 15), res=0.5)
        tx00, ty00, tz00 = tx0.flatten(), ty0.flatten(), tz0.flatten()
        indices = np.arange(bm_pix**2)
        #azs, alts = aipy.coord.top2azalt([np.array(tx0[:, :]), np.array(ty0[:, :]), np.array(tz0[:, :])])
        azs, alts = aipy.coord.top2azalt([np.array(tx0[15:16,15:16]), np.array(ty0[15:16, 15:16]), np.array(tz0[15:16, 15:16])])
        azs = np.rad2deg(azs); alts = np.rad2deg(alts)
        _sh = azs.shape
        fluxvals = np.random.random(_sh[0]) + 10
        newdata = np.zeros((2, _sh[0], _sh[1]))
        for i in range(_sh[0]):
            tx, ty, tz = aipy.coord.azalt2top([np.deg2rad(azs[i,:]), np.deg2rad(alts[i,:])])
            tx_px = tx * 0.5*bm_pix + 0.5*bm_pix
            tx_px0 = np.floor(tx_px).astype(np.int)
            tx_px1 = np.clip(tx_px0+1,0,bm_pix-1)#tx_px0 + 1
            fx = tx_px - tx_px0
            ty_px = ty * 0.5*bm_pix + 0.5*bm_pix
            ty_px0 = np.floor(ty_px).astype(np.int)
            ty_px1 = np.clip(ty_px0+1,0,bm_pix-1)#ty_px0 + 1
            fy = ty_px - ty_px0
            x0y0 = np.array([tx_px0, ty_px0], dtype=np.int)
            x0y1 = np.array([tx_px0, ty_px1], dtype=np.int)
            x1y0 = np.array([tx_px1, ty_px0], dtype=np.int)
            x1y1 = np.array([tx_px1, ty_px1], dtype=np.int)

            w0 = (1 - fx) * (1 - fy)
            w1 = fx * (1 - fy)
            w2 = (1 - fx) * fy
            w3 = fx * fy

            ps = [x0y0, x0y1, x1y0, x1y1]
            ws = [w0, w1, w2, w3]

            flip = 1
            theta = flip * np.pi / 2
            rx0y0 = rotate(theta, np.array([x0y0[0,:], x0y0[1,:]])).astype(int)
            rx0y1 = rotate(theta, np.array([x0y1[0,:], x0y1[1,:]])).astype(int)
            rx1y0 = rotate(theta, np.array([x1y0[0,:], x1y0[1,:]])).astype(int)
            rx1y1 = rotate(theta, np.array([x1y1[0,:], x1y1[1,:]])).astype(int)

            rtx_px0 = rx0y0[0, :]
            rty_px0 = rx0y0[1, :]
            rtx_px1 = rx1y1[0, :]
            rty_px1 = rx1y1[1, :]

            rps = [rx0y0, rx0y1, rx1y0, rx1y1]

            sol_dict['I%d'%i] = fluxvals[i]
            for j in range(_sh[1]):
                A_s = (bm_true[tx_px0[j], ty_px0[j]] * w0[j] + bm_true[tx_px0[j], ty_px1[j]] * w1[j] \
                  + bm_true[tx_px1[j], ty_px0[j]] * w2[j] + bm_true[tx_px1[j], ty_px1[j]] * w3[j])
                I_s = fluxvals[i] * A_s
                newdata[0, i, j] = I_s
                c = {mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j): ws[p][j] for p in xrange(len(ps))}
                eq = ' + '.join([mk_key(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]), i, j) + \
                 '*b%d'%(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j])) + '*I%d'%i for p in xrange(len(ps))])
                eqs[eq] = I_s
                consts.update(c)

                A_s = (bm_true[rtx_px0[j], rty_px0[j]] * w0[j] + bm_true[tx_px0[j], rty_px1[j]] * w1[j] \
                  + bm_true[tx_px1[j], rty_px0[j]] * w2[j] + bm_true[tx_px1[j], rty_px1[j]] * w3[j])
                I_s = fluxvals[i] * A_s
                newdata[1, i, j] = I_s
                c = {mk_key(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j]), i, j): ws[p][j] for p in xrange(len(ps))}
                eq = ' + '.join([mk_key(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j]), i, j) + \
                 '*b%d'%(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j])) + '*I%d'%i for p in xrange(len(ps))])
                eqs[eq] = I_s
                consts.update(c)

                for p in xrange(len(ps)):
                    sol_dict['b%d'%(unravel_pix(bm_pix, ps[p][0,j], ps[p][1,j]))] = 1.0
                    sol_dict['b%d'%(unravel_pix(bm_pix, rps[p][0,j], rps[p][1,j]))] = 1.0

        eqs_noise = {k:v for k,v in eqs.items()}
        ls = linsolve.LinProductSolver(eqs_noise, sol0=sol_dict, **consts)
        sol0 = ls.solve_iteratively(verbose=True)

        for key in sol0[1].keys():
            if key[0] == 'b':
                px = int(key.strip('b'))
                interp2d[px] = sol0[1].get(key)

        interp2d = interp2d.reshape((bm_pix, bm_pix))
        catd.azalt_array = np.zeros((2, _sh[0], _sh[1]))
        catd.azalt_array[0, :, :] = azs
        catd.azalt_array[1, :, :] = alts
        catd.Nsrcs = _sh[0]
        catd.Nfits = _sh[1]
        catd.data_array = newdata
        np.testing.assert_almost_equal(catd.data_array, newdata)
        np.testing.assert_almost_equal(catd.azalt_array[0, :, :], azs)
        np.testing.assert_almost_equal(catd.azalt_array[1, :, :], alts)
        bmc = bs.BeamCatCross(cat=catd, bm_pix=31)
        bmc.add_eqs(catalog_flux_xx=fluxvals, catalog_flux_yy=fluxvals, bvals=np.ones((bm_pix, bm_pix)), flip_yy=[-1])
        sol = bmc.solve()
        fluxvals, obsbeam = bmc.eval_sol(sol)
        np.testing.assert_allclose(obsbeam, interp2d)
        cleanbeam = bmc.remove_degen(bmc.ls.ls, obsbeam)
"""
