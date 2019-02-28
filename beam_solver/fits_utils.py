import numpy as np
from astropy.io import fits
from astropy import wcs
import pylab

def get_fitsinfo(fitsfile):
    """
    Extracts data and header information (metadata) from fits file
    fitsfile: Input fitsfile
    """
    data, hdr = fits.getdata(fitsfile, header=True)
    data = data.squeeze()
    freq = hdr ['CRVAL3']
    nxaxis = hdr['NAXIS1']
    nyaxis = hdr['NAXIS2']
    return {'data':data, 'hdr':hdr, 'freq':freq, 'nxaxis':nxaxis, 'nyaxis':nyaxis}

def _get_wcs(fitsfile):
    """
    Retruns the world coordinate system class
    """
    return wcs.WCS(fitsfile)

def wcs2pix(fitsfile, ra, dec):
    """
    Returns pixel numbers corresponding to ra and dec values
    fitsfile: Input fitsfile
    ra: right ascension in degrees
    dec : declination in degrees
    """
    w = _get_wcs(fitsfile)
    px1, px2 = w.all_world2pix(ra, dec, 1, 0, 0)[:2]
    px1 = np.around(px1, 0).astype(int)
    px2 = np.around(px2, 0).astype(int)
    return px1, px2

def pix2wcs(fitsfile, px1, px2):
    """
    Returns world coordinates (ra, dec) of any given pixel (px1, px2)
    fitsfile: Input fitsfile
    px1: pixel number along x-axis
    px2 : pixel number along y-axis
    """
    w = _get_wcs(fitsfile)
    ra, dec = w.all_pix2world(px1, px2, 1, 0, 0)[:2]
    return ra, dec

def add_keyword(fitsfile, key, value, outfits, overwrite=False):
    """
    Adding a keyword to the header of the fitsfile
    fitsfile: Input fitsfile
    key: Keyword to be added to the header
    value: Value of the keyword
    outfits: Output fitsfile.
    overwrite: Overwrites any exisiting file. Default is False.
    """
    _info = get_fitsinfo(fitsfile)
    hdr = _info['hdr']
    hdr.append(key)
    hdr[key] = value
    fits.writeto(outfits, _info['data'], hdr, overwrite=overwrite)

def del_keyword(fitsfile, key, outfits, overwrite=False):
    """
    Deleting keyword from the header of the fitsfile
    fitsfile: Input fitsfile
    key: Keyword to be added to the header
    outfits: Output fitsfile.
    overwrite: Overwrites any exisiting file. Default is False.
    """
    _info = get_fitsinfo(fitsfile)
    hdr = _info['hdr']
    assert key in hdr.keys(), "given key not present in the header"
    del hdr[key]
    fits.writeto(outfits, _info['data'], hdr, overwrite=overwrite)

def get_fitstats(fitsfile):
    """
    Returns statistics (min, max, std) of the fitsfile
    fitsfiles: Input fitsfile
    """
    _info = get_fitsinfo(fitsfile)
    data = _info['data']
    return {'max': np.nanmax(data), 'min': np.nanmin(data), 'std': np.nanstd(data)}

