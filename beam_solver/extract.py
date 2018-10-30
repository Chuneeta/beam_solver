"""
Functions to extract fluxes from the fits files
"""
from astropy import wcs
from astropy.io import fits
from astropy import modeling as mod
import scipy.stats as stats
import numpy as np
import pylab
import copy

def get_fitsinfo(fitsname):
    """
    Read fitsfile and extracts necessary information

    Parameters
    ----------
    fitsname : string
        Name of input fitsfile
    """

    data, header = fits.getdata(fitsname, header=True)
    data = data.squeeze()
    freq = header['CRVAL3']
    dra = header['CDELT1']
    ddec = header['CDELT2']
    nxaxis = header['NAXIS1']
    nyaxis = header['NAXIS2']

    return {'data':data, 'header':header, 'freq':freq, 'nxaxis':nxaxis, 'nyaxis':nyaxis}
    
def get_center_radec(fitsfile):
    """
    Return center ra and dec of any fitsfile

    fitsfile : string
        Name of fitsfile 
    """
    
    fitsinfo = get_fitsinfo(fitsfile)
    w = wcs.WCS(fitsinfo['hdr'])
    nxaxis = fitsinfo['nxaxis']
    nyaxis = fitsinfo['nyaxis']
    cxpix, cypix = int(naxis/2.), int(nyaxis/2.)
    cra, cdec = w.all_pix2world(cxpix, cypix, 0, 0, 0)[:2]    

    return cra, cdec

def get_flux(fitsname, ra, dec, plot=False):
    """
    Returns the peak value centered at the given ra dec coordinates within one synthesized beam

    Parameters
    ----------
    fitsname : string
        Name of input fitsfile

    ra : float
        Right ascension in degrees
    
    dec : float
        Declination in degrees

    plot : boolean
        If True, output a plot of the selected region. Default is False
    """
    fitsinfo = get_fitsinfo(fitsname)
    imdata = fitsinfo['data']
    hdr = fitsinfo['header']
    freq = fitsinfo['freq']
    nxaxis = fitsinfo['nxaxis']
    nyaxis = fitsinfo['nyaxis']
    bmaj = hdr['BMAJ']    
    bmin = hdr['BMIN']
    
    # computing synthesized beam radius and area in degrees and pixels
    w = wcs.WCS(hdr)
    dx_px = np.abs(w.wcs.cdelt[0])
    dy_px = np.abs(w.wcs.cdelt[1])
    bmaj_px = bmaj / dx_px
    bmin_px = bmin / dy_px
    bm_radius =  np.sqrt(bmaj**2 + bmin**2)
    bm_radius_px = np.sqrt(bmaj_px**2 + bmin_px**2)
    bm_area = bmaj * bmin * np.pi / 4 / np.log(2)
    px_area = dx_px * dy_px
    bm_npx = bm_area / px_area 
    ra_pix, dec_pix = w.all_world2pix(ra, dec, 0, 0, 0)[:2]
    if not np.isnan(ra_pix) and not np.isnan(dec_pix):
        ra_pix = int(ra_pix)
        dec_pix = int(dec_pix) 

    # checking if the pixel are within the range
    assert 0 <= ra_pix < nxaxis , 'Pixel value along the x-axis is outside the image range'
    assert 0 <= dec_pix < nyaxis , 'Pixel value along the y-axis is outside the image range'

    # selecting region with synthesized beam
    l_axis = np.arange(0, nxaxis)
    m_axis = np.arange(0, nyaxis)
    ll, mm = np.meshgrid(l_axis, m_axis)
        
    R = np.sqrt((ll - ra_pix)**2 + (mm - dec_pix)**2)
    select = R < bm_radius_px
    
    imdata_select = imdata[select]
    peakval = np.nanmax(imdata_select)
    minval = np.nanmin(imdata_select)
    std = np.nanstd(imdata_select)
    rms = np.std(imdata[~select]) # calculated rm outside the selected region
    peak_err = rms / np.sqrt(bm_npx / 2.0)

    peak_ind = np.argmax(imdata[select])
    peak_xpix = ll[select][peak_ind]
    peak_ypix = mm[select][peak_ind]

    # fitting 2D gaussian
    gauss_init = mod.functional_models.Gaussian2D(peakval, ra_pix, dec_pix, x_stddev=bmaj_px/2, y_stddev = bmin_px/2)
    fitter = mod.fitting.LevMarLSQFitter()
    gauss_fit = fitter(gauss_init, ll[select], mm[select], imdata_select)
   
    gauss_peak = gauss_fit.amplitude.value
    gauss_xstd = gauss_fit.x_stddev.value
    gauss_ystd = gauss_fit.y_stddev.value
    beam_theta = hdr["BPA"] * np.pi / 180
    P = np.array([ll - peak_xpix, mm - peak_ypix]).T
    Prot = P.dot(np.array([[np.cos(beam_theta), -np.sin(beam_theta)], [np.sin(beam_theta), np.cos(beam_theta)]]))
    gauss_cov = np.array([[gauss_xstd**2, 0], [0, gauss_ystd**2]])
    model_gauss = stats.multivariate_normal.pdf(Prot, mean=np.array([0,0]), cov=gauss_cov)
    model_gauss *= gauss_fit.amplitude.value / model_gauss.max()
    gauss_int = np.nansum(model_gauss) / bm_npx

    if plot:
        imdata_copy = copy.deepcopy(imdata)
        imdata_copy[~select] = np.nan

        fig = pylab.figure(figsize=(6, 4))
        my_wcs = wcs.WCS(hdr, naxis=[wcs.WCSSUB_CELESTIAL])
        ax = pylab.subplot(121, projection=my_wcs)
        pylab.imshow(imdata_copy, aspect='auto', cmap='jet', origin='lower')
        pylab.grid(color='black')
        pylab.colorbar()

        ax = pylab.subplot(122, projection=my_wcs)
        pylab.imshow(model_gauss.T, aspect='auto', cmap='jet', origin='lower')
        pylab.grid(color='black')
        pylab.colorbar()
        pylab.show()

    return {'freq':freq, 'peak':peakval, 'min':minval, 'peak_err':peak_err, 'gauss_peak':gauss_peak, 'gauss_int':gauss_int}

