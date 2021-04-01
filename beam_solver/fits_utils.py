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
    Returns the world coordinate system class
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

def write_fits(fitsfile, data, outfile, overwrite=False):
    """
    Overwrite exisiting file with new data keeping the old metadata
    fitsfile: Input fitsfile
    data: Ndarray containing the new values, can be float or ints
    outfile: name output fitsfile
    overwrite: Overwrite is set to False
    """
    _info = get_fitsinfo(fitsfile)
    fits.writeto(outfile, data, _info['hdr'], overwrite=overwrite)

def calc_solint(fitsfile, nants, inttime):
    """
    Return solution interval used for self-caibration, should be greater than the
    interval returned by the function
    fitsfile: Name of input fitsfile
    nants :  Number of antennas in the array layout
    inttime: Integration in seconds of the observation
    """
    stats_dict = get_fitstats(fitsfile)
    mxval, mnval, std = stats_dict['max'], stats_dict['min'], stats_dict['std']
    solint = (std / mxval)**2 * (inttime * (nants - 3)) / 9
    return solint

def plot_fitsfile(fitsfile, cmap='gray', vmin=None, vmax=None, savefig=False, figname=''):
    _info = get_fitsinfo(fitsfile)
    hdr = _info['hdr']
    data = _info['data']
    if vmin is None: vmin = np.min(data)
    if vmax is None: vmax = np.max(data)
    my_wcs = wcs.WCS(hdr, naxis=[wcs.WCSSUB_CELESTIAL])
    fig = pylab.figure()
    ax = fig.add_subplot(111, projection=my_wcs)
    im = ax.imshow(data.squeeze(), aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = pylab.colorbar(im, ax=ax)
    ax.grid(ls='dotted', color='white')
    ax.coords[0].set_axislabel('RA (deg)')
    ax.coords[1].set_axislabel('DEC (deg)')
    if savefig:
        pylab.savefig(figname)
        pylab.close()
    else:
        pylab.show()
