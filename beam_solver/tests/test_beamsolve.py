from beam_solver.data import DATA_PATH
from collections import OrderedDict
from beam_solver import catdata as cd
from beam_solver import beamsolve as bs
from beam_solver import beam_utils as bt
from beam_solver import fits_utils as ft
import nose.tools as nt
import numpy as np
import collections
import aipy
import os
import copy
import pylab
import linsolve

beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
fitsfile = os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')
outfile = fitsfile.replace('.fits', '.mod.fits')
ras = [74.26237654, 41.91116875, 22.47460079, 9.8393989, 356.25426296]
decs = [-52.0209015, -43.2292595, -30.27372862, -17.40763737, -0.3692835] 

ft.add_keyword(fitsfile, 'JD', 2458115.23736, outfile, overwrite=True)
catd = cd.catData()
catd.gen_catalog(ras, decs, [outfile])

def create_catdata(azalt, data, error, nsrcs, npoints):
    catd = cd.catData()
    catd.azalt_array = azalt
    catd.data_array = data
    catd.error_array = error
    catd.Nfits = npoints
    catd.Nsrcs = nsrcs
    return catd

def gen_catdata_zensrc(fluxval, sigma=1):
    azs = np.array([[np.pi/2]]); alts=np.array([[np.pi/2]])
    top = aipy.coord.azalt2top([azs, alts])
    data = bt.get_src_tracks(top, fluxval, sigma_x=sigma)
    data = data.reshape((1, 1, 1))
    catd = create_catdata(np.array([azs, alts]), data, 0.1*data, 1, 1)
    return catd

def gen_random_track(npoints, fluxval, sigma=1):
    azs = np.random.uniform(np.pi/2, 3*np.pi/2, npoints)
    azs = np.sort(azs)
    alts = np.random.uniform(0, np.pi/2, npoints/2)
    alts = np.sort(alts)
    alts = np.append(alts[:-1], np.append(alts[::-1], 0))
    top = aipy.coord.azalt2top([azs, alts])
    data = bt.get_src_tracks(top, fluxval, sigma_x=sigma)
    return np.array([azs, alts]), data

def gen_catdata_nsrcs(nsrcs, npoints,fluxvals, sigma=1):
    data_arr = np.zeros((1, nsrcs, npoints))
    azalt_arr = np.zeros((2, nsrcs, npoints))
    for ii in range(nsrcs):
        azalt, data = gen_random_track(npoints, fluxvals[ii], sigma)
        data_arr[0, ii, :] = data
        azalt_arr[0, ii, :] = azalt[0]
        azalt_arr[1, ii, :] = azalt[1]
    catd = create_catdata(azalt_arr, data_arr, 0.1*data, nsrcs, npoints)
    return catd

def gen_catdata_grid(npix, fluxvals, sigma_x, sigma_y=None):
    if sigma_y is None: sigma_y = sigma_x
    tx, ty, tz = bt.get_top(npix, center=(int(npix/2), int(npix/2)), res=1)
    azs, alts = aipy.coord.top2azalt(np.array([tx, ty, tz]))
    data_arr = np.zeros((1, npix, npix))
    for ii in range(npix):
        data_arr[0, ii, :] = bt.get_src_tracks(np.array([tx[ii,: ], ty[ii, :], tz[ii, :]]), fluxvals[ii], sigma_x=sigma_x, sigma_y=sigma_y)    
    catd = create_catdata(np.array([azs, alts]), data_arr, 0.1*data_arr, npix, npix)
    return catd

class Test_BeamOnly():
    def test_init(self):
        catd = cd.catData()
        bms = bs.BeamOnly(catd)
        nt.assert_equal(bms.bm_pix, 61)
        nt.assert_equal(bms.cat, catd)
        bms = bs.BeamOnly(catd, 30)
        nt.assert_equal(bms.bm_pix, 30)
        nt.assert_equal(bms.cat, catd)

    def test_mk_key(self):
        bms = bs.BeamOnly()
        key = bms._mk_key(0, 0, 0)
        nt.assert_equal(key, 'w0_s0_t0')
        key = bms._mk_key(0, 1, 4)
        nt.assert_equal(key, 'w0_s1_t4')
       
    def test_unravel_pix(self):
        bms = bs.BeamOnly()
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
        bms = bs.BeamOnly(bm_pix=31)
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
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        nt.assert_equal(len(ps), 4)
        nt.assert_equal(len(ws), 4)
        np.testing.assert_equal(ps, [np.array([[15], [15]]), np.array([[15], [16]]), np.array([[16], [15]]), np.array([[16], [16]])]) 
        nt.assert_almost_equal(np.sum(ws), 1.0)
        np.testing.assert_almost_equal(ws, [np.array([1.0]), np.array([0.]), np.array([0.]), np.array([0.])])

    def test_rotate_ps(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 2*np.pi, 1)
        np.testing.assert_equal(ps, [np.array([[15], [15]]), np.array([[15], [16]]), np.array([[16], [15]]), np.array([[16], [16]])])
        bms = bs.BeamOnly(catd, 31)
        ps, ws = bms.get_weights(np.array([[np.pi/4], [np.pi/4]]), np.pi/2, 1)
        ps_270, ws_270 = bms.get_weights(np.array([[np.pi/4], [np.pi/4]]), -3 * np.pi/2, 1)
        np.testing.assert_almost_equal(ps, ps_270)
        np.testing.assert_almost_equal(ws, ws_270)

    def test_flip_ps(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, -1)
        np.testing.assert_equal(ps, [np.array([[15], [15]]), np.array([[15], [16]]), np.array([[16], [15]]), np.array([[16], [16]])])            
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/4], [np.pi/4]]), 0, 1)
        ps_f, ws_f = bms.get_weights(np.array([[np.pi/4], [np.pi/4]]), 0, -1)
        for i in range(len(ps)):
            nt.assert_almost_equal(ps_f[i][0], ps[i][0] - 15)

    def test_mk_eq(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=True)
        eq_keys = bms.eqs.keys()
        cns_keys = bms.consts.keys()
        nt.assert_almost_equal(len(eq_keys), 1)
        px0 = bms.unravel_pix(31, (15, 15))
        px1 = bms.unravel_pix(31, (15, 16))
        px2 = bms.unravel_pix(31, (16, 15))
        px3 = bms.unravel_pix(31, (16, 16)) 
        eq0 = 'w%s_s0_t0*b%s + w%s_s0_t0*b%s + w%s_s0_t0*b%s + w%s_s0_t0*b%s'%(px0, px0, px1, px1, px2, px2, px3, px3)
        nt.assert_equal(list(eq_keys)[0], eq0)
    
    def test_eqs(self):
        bms = bs.BeamOnly(catd, 31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=True)
        eq_keys = bms.eqs.keys()
        nt.assert_almost_equal(bms.eqs[list(eq_keys)[0]], 1)

    def test_consts(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=True)
        cns_keys = bms.consts.keys()
        nt.assert_almost_equal(len(cns_keys), 4)
        px0 = bms.unravel_pix(31, (15, 15))
        nt.assert_almost_equal(bms.consts['w%s_s0_t0'%px0], 1.0)   
 
    def test_calc_catalog_flux(self):
        bms = bs.BeamOnly(catd, 31)
        beam = bt.get_fitsbeam(beamfits, 151e6)
        catalog_flux = bms.calc_catalog_flux(beam, 'xx')
        nt.assert_almost_equal(catalog_flux[2], 1.000, 3)

    def test_build_solver(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=True)
        bms._build_solver()
        nt.assert_true(isinstance(bms.ls, linsolve.LinearSolver))

    def test_get_A(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=True)
        bms._build_solver()
        A = bms.get_A(bms.ls)
        np.testing.assert_almost_equal(A, np.array([[[1.], [0.], [0.], [0.]]]))

    def test_svd(self):
        bms = bs.BeamOnly(bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=True)
        bms._build_solver()
        A = bms.get_A(bms.ls)
        U, S, V = bms.svd(bms.ls, A)
        nt.assert_almost_equal(S[0], 1.0)
        np.testing.assert_almost_equal(np.dot(U, U.T), np.identity(4))
        np.testing.assert_almost_equal(np.dot(V, V.T), np.identity(4))

    def test_remove_degen(self):
        pass

    def test_add_eqs(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval)
        eq_keys = bms.eqs.keys()
        nt.assert_equal(len(eq_keys), 1)

    def test_eval_sol(self):
        sol = {'b1': 1}
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        output = bms.eval_sol(sol)
        answer = np.zeros((31, 31))
        answer.flat[1] = 1
        np.testing.assert_equal(output, answer)
        sol = {'b%d'%i:1 for i in range(31**2)}
        output = bms.eval_sol(sol)
        answer = np.ones((31, 31))
        np.testing.assert_equal(output, answer)

    def test_sol_keys(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval)
        sol = bms.solve()
        nt.assert_equal(len(sol.keys()), 4)
        nt.assert_true('b{}'.format(bms.unravel_pix(31, (15, 15))) in sol.keys())
        nt.assert_true('b{}'.format(bms.unravel_pix(31, (15, 16))) in sol.keys())
        nt.assert_true('b{}'.format(bms.unravel_pix(31, (16, 15))) in sol.keys())
        nt.assert_true('b{}'.format(bms.unravel_pix(31, (16, 16))) in sol.keys())

    def test_sol_values(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval)
        sol = bms.solve()
        nt.assert_true(sol['b{}'.format(bms.unravel_pix(31, (15, 15)))], 1.0) 

    def test_solve_src(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval)
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)       
        answer = np.zeros((31, 31))
        answer[15, 15] = 1
        np.testing.assert_almost_equal(obsbeam, answer)
    
    def test_solve_grid(self):
        fluxvals = np.random.random(50) + 10
        catd = gen_catdata_grid(50, fluxvals, sigma_x=0.2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxvals)
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)
        gaussbeam = bt.get_gaussbeam(3, mu_x=15, mu_y=15, size=31)
        np.testing.assert_allclose(gaussbeam, obsbeam, atol=1e-01) # most values are equal to 1e-2, with some exceptions

    def test_beam_rotate(self):
        fluxvals = np.random.random(50) + 10
        catd = gen_catdata_grid(50, fluxvals, sigma_x=0.4, sigma_y=0.2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxvals)
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxvals, theta=[np.pi/2])
        sol = bms.solve()
        obsbeam_90 = bms.eval_sol(sol)
        np.testing.assert_allclose(obsbeam, np.rot90(obsbeam_90), atol=1e-01)

    def test_diag_noise(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval)
        nt.assert_equal(len(bms.diag_noise), 1)

    def test_noise_matrix(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamOnly(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval)
        noise_array = bms.get_noise_matrix()
        nt.assert_equal(noise_array.shape, (1, 1))

class Test_BeamCat():
    def test_mk_eq(self):
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, equal_wgts=False, bvals=np.ones((31, 31)))
        eq_keys = bms.eqs.keys()
        cns_keys = bms.consts.keys()
        nt.assert_almost_equal(len(eq_keys), 1)
        px0 = bms.unravel_pix(31, (15, 15))
        px1 = bms.unravel_pix(31, (15, 16))
        px2 = bms.unravel_pix(31, (16, 15))
        px3 = bms.unravel_pix(31, (16, 16))
        eq0 = 'w%s_s0_t0*b%s*I0 + w%s_s0_t0*b%s*I0 + w%s_s0_t0*b%s*I0 + w%s_s0_t0*b%s*I0'%(px0, px0, px1, px1, px2, px2, px3, px3)
        nt.assert_equal(list(eq_keys)[0], eq0)

    def test_consts(self):
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        ps, ws = bms.get_weights(np.array([[np.pi/2], [np.pi/2]]), 0, 1)
        bms._mk_eq(ps, ws, 1, 1, 0, 0, bvals=np.ones((31, 31)), equal_wgts=False)
        cns_keys = bms.consts.keys()
        nt.assert_almost_equal(len(cns_keys), 4)
        px0 = bms.unravel_pix(31, (15, 15))
        nt.assert_almost_equal(bms.consts['w%s_s0_t0'%px0], 1.0)

    def test_add_eqs(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.ones((31, 31)))
        eq_keys = bms.eqs.keys()
        nt.assert_equal(len(eq_keys), 1)

    def test_eqs(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.ones((31, 31)))
        eq_keys = list(bms.eqs.keys())
        nt.assert_almost_equal(bms.eqs[eq_keys[0]], 2.0)

    def test_add_constrain(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.ones((31, 31)))
        bms.add_constrain(0, 20)
        keys = list(bms.eqs.keys())
        nt.assert_equal(len(keys), 2)
        nt.assert_equal(bms.eqs[keys[1]], 20)

    def test_diag_noise(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.ones((31, 31)))     
        nt.assert_equal(len(bms.diag_noise), 1)

    def test_noise_matrix(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.ones((31, 31)))
        noise_array = bms.get_noise_matrix()
        nt.assert_equal(noise_array.shape, (1, 1))
    
    def test_build_solver(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.ones((31, 31)))
        bms._build_solver(norm_weight=100)
        nt.assert_true(isinstance(bms.ls, linsolve.LinProductSolver))

    def test_sol_values(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.zeros((31, 31)))
        sol = bms.solve()
        nt.assert_true(sol[1]['b{}'.format(bms.unravel_pix(31, (15, 15)))], 1.0)

    def test_eval_sol(self):
        sol = ({'chisq': 0.0, 'conv_crit': 0.0, 'iter': 1}, {'b1': 1, 'I0':2})
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        fluxval, obsbeam = bms.eval_sol(sol)
        ansbeam = np.zeros((31, 31))
        ansbeam.flat[1] = 1
        np.testing.assert_equal(obsbeam, ansbeam)
        np.testing.assert_almost_equal(fluxval, np.array([[1], [2.0]]))
        sol0 = {'b%d'%i:1 for i in range(31**2)}
        sol0['I0'] = 2 
        sol = ({'chisq': 0.0, 'conv_crit': 0.0, 'iter': 1}, sol0)
        fluxval, obsbeam = bms.eval_sol(sol)
        ansbeam = np.ones((31, 31))
        np.testing.assert_almost_equal(obsbeam, ansbeam)
        np.testing.assert_almost_equal(fluxval, np.array([[1], [2.0]]))

    def test_solve_src(self):    
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        bms = bs.BeamCat(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux=fluxval, bvals=np.zeros((31, 31)))
        sol = bms.solve()
        outflux, obsbeam = bms.eval_sol(sol)
        ansbeam = np.zeros((31, 31))
        ansbeam[15, 15] = 1
        np.testing.assert_almost_equal(obsbeam, ansbeam)
        np.testing.assert_almost_equal(outflux, np.array([[1], [2.0]])) 

class Test_BeamOnlyCross():
    def test_solver(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        newdata = np.zeros((2, 1, 1))
        newdata[:, :, :] = catd.data_array[0, :, :]
        catd.data_array = newdata
        catd.error_array = 0.1 * newdata
        catd.Npols = 2
        bms = bs.BeamOnlyCross(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux_xx=fluxval, catalog_flux_yy=fluxval, theta_xx=[0], theta_yy=[np.pi/2])
        sol = bms.solve()
        nt.assert_true(isinstance(bms.ls, linsolve.LinearSolver))

    def test_solve_onesrc(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        newdata = np.zeros((2, 1, 1))
        newdata[:, :, :] = catd.data_array[0, :, :]
        catd.data_array = newdata
        catd.error_array = 0.1 * newdata
        catd.Npols = 2
        bms = bs.BeamOnlyCross(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux_xx=fluxval, catalog_flux_yy=fluxval, theta_xx=[0], theta_yy=[np.pi/2])
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)
        ansbeam = np.zeros((31, 31))
        ansbeam[15, 15] = 1
        np.testing.assert_almost_equal(obsbeam, ansbeam)

    def test_solve_grid(self):
        fluxvals = np.random.random(50) + 10
        catd= gen_catdata_grid(50, fluxvals, sigma_x=0.2, sigma_y=0.2)
        newdata = np.zeros((2, 50, 50))
        newdata[:, :, :] = catd.data_array[0, :, :]
        catd.data_array = newdata
        catd.error_array = 0.1 * newdata
        catd.Npols = 2
        bms = bs.BeamOnlyCross(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux_xx=fluxvals, catalog_flux_yy=fluxvals, theta_xx=[0], theta_yy=[np.pi/2], flip_yy=[-1])
        sol = bms.solve()
        obsbeam = bms.eval_sol(sol)
        gaussbeam = bt.get_gaussbeam(3, mu_x=15, mu_y=15, size=31)
        np.testing.assert_allclose(gaussbeam, obsbeam, atol=1e-01)

class Test_BeamCatCross():
    def test_solver(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        newdata = np.zeros((2, 1, 1))
        newdata[:, :, :] = catd.data_array[0, :, :]
        catd.data_array = newdata
        catd.error_array = 0.1 * newdata
        catd.Npols = 2
        bms = bs.BeamCatCross(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux_xx=fluxval, catalog_flux_yy=fluxval, theta_xx=[0], theta_yy=[np.pi/2], bvals=np.zeros((31, 31)))
        sol = bms.solve(norm_weight=100)
        nt.assert_true(isinstance(bms.ls, linsolve.LinProductSolver))

    def test_solve_onesrc(self):
        fluxval = np.array([2.0])
        catd = gen_catdata_zensrc(fluxval, sigma=2)
        newdata = np.zeros((2, 1, 1))
        newdata[:, :, :] = catd.data_array[0, :, :]
        catd.data_array = newdata
        catd.error_array = 0.1 * newdata
        catd.Npols = 2
        bms = bs.BeamCatCross(cat=catd, bm_pix=31)
        bms.add_eqs(catalog_flux_xx=fluxval, catalog_flux_yy=fluxval, theta_xx=[0], theta_yy=[np.pi/2], bvals=np.zeros((31, 31)))
        sol = bms.solve()
        fluxvals, obsbeam = bms.eval_sol(sol)
        ansbeam = np.zeros((31, 31))
        ansbeam[15, 15] = 1
        np.testing.assert_almost_equal(obsbeam, ansbeam)
        np.testing.assert_almost_equal(fluxvals[1, :], np.array([2.0]))
