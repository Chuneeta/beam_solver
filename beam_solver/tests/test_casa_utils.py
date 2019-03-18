import numpy as np
import nose.tools as nt
from beam_solver import casa_utils as ct
from beam_solver.data import DATA_PATH
import pyuvdata
import os

uvfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
uvfits = uvfile + '.uvfits'
msfile = uvfile + '.ms'
imagename = os.path.join(DATA_PATH, '2457698.40355.xx')
fitsname = imagename + '.fits'

def test_uvfits2ms():
    uvd = pyuvdata.UVData()
    uvd.read_miriad(uvfile)
    uvd.write_uvfits(uvfits, force_phase=True, spoof_nonessential=True)
    ct.uvfits2ms(uvfits, script='uvfits2ms', delete=True)
    nt.assert_true(os.path.exists(msfile))

def test_ms2uvfits():
    os.system('rm -rf {}'.format(uvfits))
    ct.ms2uvfits(msfile, outfile=uvfits, script='ms2uvfits', delete=False)
    nt.assert_true(os.path.exists(uvfits))

def test_flag_antenna():
    pass

def test_imaging():
    ct.imaging(msfile, imagename, antenna='', cellsize='8arcmin', npix=512, niter=0, threshold='0Jy', weighting='uniform', start=200, stop=900, uvlength=0, script='clean', delete=True)
    nt.assert_true(os.path.exists(imagename + '.image'))
    nt.assert_true(os.path.exists(imagename + '.psf'))
    nt.assert_true(os.path.exists(imagename + '.model'))

def test_exportfits():
    ct.imaging(msfile, imagename)
    ct.exportfits(imagename + '.image', fitsname)
    nt.assert_true(os.path.exists(fitsname))

    # removing created files
    os.system('rm -rf *.log')
    os.system('rm -rf *.last')
    os.system('rm -rf {}'.format(uvfits))
    os.system('rm -rf {}'.format(msfile))
    os.system('rm -rf {}.model'.format(imagename))
    os.system('rm -rf {}.residual'.format(imagename))
    os.system('rm -rf {}.psf'.format(imagename))
    os.system('rm -rf {}.flux'.format(imagename))
    os.system('rm -rf {}.image'.format(imagename))
    os.system('rm -rf {}'.format(fitsname))

def test_generate_complist_input():
    ct.generate_complist_input([30.69], [-30.75], [-1], [0], [151], output='complist.dat')
    cmplist = np.loadtxt('complist.dat', dtype='str', delimiter=':')
    nt.assert_almost_equal(cmplist[0], 'J2000 2h2m45.6s -30d45m0.0s')
    os.system('rm -rf complist.dat')

def test_create_complist():
    ct.generate_complist_input([30.69], [-30.75], [-1], [0], [151], output='complist.dat')    
    ct.create_complist('complist.dat', 'component.cl')
    nt.assert_true(os.path.exists('component.cl'))
    os.system('rm -rf complist.dat')

def test_ft():
    pass

