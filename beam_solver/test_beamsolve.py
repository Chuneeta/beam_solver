from beam_solver.data import DATA_PATH
import catdata as cd
import beamsolve as bs
import beam_utils as bt
import nose.tools as nt
import numpy as np
import aipy
import os
import copy
import pylab

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

# generating catalog
catd = cd.catData()
catd.gen_catalog(fitsfiles_xx, ras, decs)

class Test_BeamSolveBase():
    def test_init(self):
        bms = bs.BeamSolveBase(catd)
        nt.assert_equal(bms.bm_pix, 60)
        nt.assert_equal(bms.cat, catd)
        bms = bs.BeamSolveBase(catd, 30)
        nt.assert_equal(bms.bm_pix, 30)
        nt.assert_equal(bms.cat, catd)

    def test_mk_key(self):
        bms = bs.BeamSolveBase(catd)
        key = bms._mk_key(0, 0, 0)
        nt.assert_equal(key, 'w0_s0_t0')
        key = bms._mk_key(0, 1, 4)
        nt.assert_equal(key, 'w0_s1_t4')

    def test_unravel_pix(self):
        bms = bs.BeamSolveBase(catd)
        ind = bms.unravel_pix(60, (0, 0))
        nt.assert_equal(ind, 0)
        ind = bms.unravel_pix(60, (0, 5))
        nt.assert_equal(ind, 5)    
        bms = bs.BeamSolveBase(catd, bm_pix=30)
        ind = bms.unravel_pix(30, (0, 5))
        nt.assert_equal(ind, 5)
        ind = bms.unravel_pix(30, (2, 5))
        nt.assert_equal(ind, 65)

    def test_rotate_mat(self):
        bms = bs.BeamSolveBase(catd)
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

    def test_get_weights(self):
        bms = bs.BeamSolveBase(catd, 4)
        ps, ws = bms.get_weights(np.array([[0], [0]]))
        bms = bs.BeamSolveBase(catd, 12)
        ps, ws = bms.get_weights(np.array([[0], [0]]))
        nt.assert_equal(ps[0][0][0], ps[1][0][0])
        nt.assert_equal(ps[2][0][0], ps[3][0][0])
        nt.assert_equal(ps[0][1][0], ps[2][1][0])
        nt.assert_equal(ps[1][1][0], ps[3][1][0])
        nt.assert_almost_equal(np.sum(ws), 1)
        bms = bs.BeamSolveBase(catd)
        ps, ws = bms.get_weights(np.array([[0], [0]]))
        nt.assert_equal(ps[0][0][0], ps[1][0][0])
        nt.assert_equal(ps[2][0][0], ps[3][0][0])
        nt.assert_equal(ps[0][1][0], ps[2][1][0])
        nt.assert_equal(ps[1][1][0], ps[3][1][0])
        nt.assert_almost_equal(np.sum(ws), 1)

    def test_solve(self):
        beamxx = bt.get_fitsbeam(beamfits, 151e6)
        catalog_flux = catd.calc_catalog_flux(beamxx, 'xx')
        # simulating using a gaussian beam
        catd_copy = copy.deepcopy(catd)
        azalts = catd_copy.azalt_array
        bm_pix = 31
        bm_true = np.zeros((bm_pix, bm_pix), dtype=float)
        bmx, bmy = np.indices(bm_true.shape)
        mu = 15.; sigma=3.
        bm_true = np.exp(-((bmx-mu)**2 + (bmy-mu)**2)/ (2 * sigma**2))
        tx0, ty0, tz0 = get_top(bm_pix, center=(15, 15), res=0.5)
        tx00, ty00, tz00 = tx0.flatten(), ty0.flatten(), tz0.flatten()
        indices = np.arange(bm_pix**2)
        azs, alts = aipy.coord.top2azalt([np.array(tx0[:,:]), np.array(ty0[:,:]), np.array(tz0[:,:])])
        azs = np.rad2deg(azs); alts= np.rad2deg(alts)
        _sh = azs.shape
        fluxvals = np.random.random(_sh[1]) + 10
        newdata = np.zeros((1, _sh[0], _sh[1]))
        catd_copy.azalt_array = np.zeros((2, _sh[0], _sh[1]))
        catd_copy.azalt_array[0, :, :] = azs
        catd_copy.azalt_array[1, :, :] = alts
        for i in range(_sh[0]):
            tx, ty, tz = aipy.coord.azalt2top([np.deg2rad(catd_copy.azalt_array[0, i, :]), np.deg2rad(catd_copy.azalt_array[1, i, :])])
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
    
            w0 = ((1 - fx) * (1 - fy))**(-1)
            w1 = (fx * (1 - fy))**(-1)
            w2 = ((1 - fx) * fy)**(-1)
            w3 = (fx * fy)**(-1)

            for j in range(_sh[1]):
                weights = w0[j] + w1[j] + w2[j] + w3[j]
                A_s = (bm_true[tx_px0[j],ty_px0[j]] * w0[j] + bm_true[tx_px0[j],ty_px1[j]] * w1[j] + bm_true[tx_px1[j],ty_px0[j]] * w2[j] + bm_true[tx_px1[j], ty_px1[j]] * w3[j])/weights
                newdata[0, i, j] = fluxvals[i] * A_s
        catd_copy.data_array = newdata 
        np.testing.assert_almost_equal(catd_copy.data_array, newdata)
        np.testing.assert_almost_equal(catd_copy.azalt_array[0], azs)
        np.testing.assert_almost_equal(catd_copy.azalt_array[1], alts)
        bms = bs.BeamSolve(cat=catd_copy, bm_pix=31)
        bmss = bms.beamsolver()
	bmss.construct_linear_sys(catalog_flux=fluxvals)
	sol = bmss.ls.solve(verbose=True)
	obsbeam = bmss.eval_sol(sol)
        #np.testing.assert_almost_equal(bm_true, obsbeam) 
 
class Test_BeamSolveCross():
    def test_init(self):
        bms = bs.BeamSolveCross(catd)
        nt.assert_equal(bms.bm_pix, 60)
        nt.assert_equal(bms.cat, catd)
        bms = bs.BeamSolveBase(catd, 30)
        nt.assert_equal(bms.bm_pix, 30)
        nt.assert_equal(bms.cat, catd)

