import nose.tools as nt
from beam_solver import fits_utils as ft
from beam_solver.data import DATA_PATH
import numpy as np
import os

fitsfile = os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')
outfits = fitsfile.replace('.fits', '.mod.fits')

def test_get_fitsinfo():
    _info = ft.get_fitsinfo(fitsfile)
    nt.assert_true(type(_info['hdr']), "<class 'astropy.io.fits.header.Header'>")
    nt.assert_equal(_info['data'].shape, (512, 512))
    nt.assert_equal(_info['nxaxis'], 512)
    nt.assert_equal(_info['nyaxis'], 512)
    nt.assert_equal(_info['freq'], 1.537109425036e8)

def test_get_wcs():
    w = ft._get_wcs(fitsfile)
    nt.assert_true(type(w), 'astropy.wcs.wcs.WCS')

def test_wcs2pix():
    px1, px2 = ft.wcs2pix(fitsfile, 23.09216344506, -30.80854184914)
    nt.assert_almost_equal(px1, 256)
    nt.assert_almost_equal(px2, 256)
    px1, px2 = ft.wcs2pix(fitsfile, [ 98.27839626, 23.09216345, 345.50771846], [-51.95930478, -30.80854185, 13.36480705])
    np.testing.assert_almost_equal(px1, [0, 256, 511])
    np.testing.assert_almost_equal(px2, [0, 256, 511]) 

def test_pix2wcs():
    ra, dec = ft.pix2wcs(fitsfile, 256, 256)
    nt.assert_almost_equal(ra, 23.09216345)
    nt.assert_almost_equal(dec, -30.80854185)
    ra, dec = ft.pix2wcs(fitsfile, [0, 256, 511], [0, 256, 511])
    np.testing.assert_almost_equal(ra, [ 98.27839626, 23.09216345, 345.50771846])
    np.testing.assert_almost_equal(dec, [-51.95930478, -30.80854185, 13.36480705])

def test_add_keyword():
    ft.add_keyword(fitsfile, 'JD', 2458115.23736, outfits, overwrite=True)
    nt.assert_true(os.path.exists(outfits)) 
    _info = ft.get_fitsinfo(outfits)
    nt.assert_true('JD' in _info['hdr'].keys())
    nt.assert_equal(_info['hdr']['JD'], 2458115.23736)

def test_del_keyword():
    ft.del_keyword(outfits, 'JD', outfits, overwrite=True)
    _info = ft.get_fitsinfo(outfits)
    nt.assert_false('JD' in _info['hdr'].keys())

def test_no_keyword():
    nt.assert_raises(AssertionError, ft.del_keyword, outfits, 'JD', outfits)
    
def test_get_fitstats():
    stats = ft.get_fitstats(fitsfile)
    nt.assert_equal(stats['min'], 0.0)
    nt.assert_equal(stats['max'], 1.0)
    nt.assert_almost_equal(stats['std'], 0.016148191)
    
    #deleting created files
    os.system('rm -rf {}'.format(outfits))
