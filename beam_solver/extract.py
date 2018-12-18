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
import warnings

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
    
def get_centre_radec(fitsfile):
    """
    Return center ra and dec of any fitsfile

    Parameters
    ----------	
    fitsfile : string
        Name of fitsfile 
    """
    
    fitsinfo = get_fitsinfo(fitsfile)
    hdr = fitsinfo['header']
    w = wcs.WCS(hdr)
    cxpix, cypix = int(hdr['CRPIX1']), int(hdr['CRPIX2'])
    cra, cdec = w.all_pix2world(cxpix, cypix, 0, 0, 0)[:2]

    return cra, cdec

def get_flux(fitsname, ra, dec, flux_type='peak'):
    """
    Returns the peak value centered at the given ra dec coordinates within one synthesized beam

    Parameters
    ----------
    fitsname : string
        Name of input fitsfile.

    ra : float
        Right ascension in degrees.
    
    dec : float
        Declination in degrees.

    flux_type : str
        Type of the flux density to return, can be eithe 'peak' or 'total'.
        'peak' return the peak pixel value selected from all the pixels
        confined within the synthesized beam. 'total' returns the integrated
        flux density from a gaussian fit around the source.
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

    if (0 <= ra_pix < nxaxis) and ( 0 <= dec_pix < nyaxis):
        # selecting region with synthesized beam
        l_axis = np.arange(0, nxaxis)
        m_axis = np.arange(0, nyaxis)
        ll, mm = np.meshgrid(l_axis, m_axis)
        
        R = np.sqrt((ll - ra_pix)**2 + (mm - dec_pix)**2)
        select = R < 2 * bm_radius_px
    
        imdata_select = imdata[select]
        peakval = np.nanmax(imdata_select)
        minval = np.nanmin(imdata_select)
        std = np.nanstd(imdata_select)
        rms = np.sqrt(np.nanmean(imdata[~select])**2) # calculated rms outside the selected region
        peak_err = rms / np.sqrt(bm_npx / 2.0)

        if flux_type == 'total':
            peak_ind = np.argmax(imdata[select])
            peak_xpix = ll[select][peak_ind]
            peak_ypix = mm[select][peak_ind]

            # fitting 2D gaussian
            gauss_init = mod.functional_models.Gaussian2D(peakval, peak_xpix, peak_ypix, x_stddev=bmaj_px/2., y_stddev = bmin_px/2.)
            fitter = mod.fitting.LevMarLSQFitter()
            gauss_fit = fitter(gauss_init, ll[select], mm[select], imdata_select)
   
            gauss_peak = gauss_fit.amplitude.value
            gauss_xstd = gauss_fit.x_stddev.value
            gauss_ystd = gauss_fit.y_stddev.value
            beam_theta = hdr["BPA"] * np.pi / 180
            P = np.array([ll - peak_xpix, mm - peak_ypix]).T
            Prot = P.dot(np.array([[np.cos(beam_theta), -np.sin(beam_theta)], [np.sin(beam_theta), np.cos(beam_theta)]]))
            gauss_cov = np.array([[gauss_xstd**2, 0], [0, gauss_ystd**2]])

            try:
                model_gauss = stats.multivariate_normal.pdf(Prot, mean=np.array([0,0]), cov=gauss_cov)
                model_gauss *= gauss_fit.amplitude.value / model_gauss.max()
                flux = np.nansum(model_gauss) / bm_npx
                err = peak_err # need to check for another metric

            except:
                flux, err = np.nan        

        elif flux_type == 'peak':
            flux, err = peakval, peak_err

        else:
           raise ValueError('flux_type is not recognized.')
        
    else:
        warnings.warn('WARNING: Right ascension or declination outside image field, therefore values are set to nan', Warning)
        flux, err = np.nan, np.nan   
 
    return {'freq': freq, 'flux': flux, 'error': err} 

