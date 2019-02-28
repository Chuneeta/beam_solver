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
    pass

def test_create_complist():
    pass

def test_ft():
    pass

