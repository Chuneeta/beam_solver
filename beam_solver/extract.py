import numpy as np
from beam_solver import fits_utils as ft
from beam_solver.data import DATA_PATH
import warnings
from astropy.modeling import models, fitting
import pylab
import copy

def get_flux(fitsfile, ra, dec):
    """
    Returns the statistics obtained from the desired or selected region
    fitsfile: Input fitsfile
    ra : Right ascension in degrees
    dec : Declination in degrees
    """
    fitsinfo = ft.get_fitsinfo(fitsfile)
    imdata = fitsinfo['data']
    hdr = fitsinfo['hdr']
    freq = fitsinfo['freq']
    nxaxis = fitsinfo['nxaxis']
    nyaxis = fitsinfo['nyaxis']
    bmaj = hdr['BMAJ']
    bmin = hdr['BMIN']
    bpa = hdr['BPA']
    # computing synthesized beam radius and area in degrees and pixels
    w = ft._get_wcs(fitsfile)
    dx_px = np.abs(w.wcs.cdelt[0])
    dy_px = np.abs(w.wcs.cdelt[1])
    bmaj_px = bmaj / dx_px
    bmin_px = bmin / dy_px
    bm_radius =  np.sqrt(bmaj**2 + bmin**2)
    bm_radius_px = np.sqrt(bmaj_px**2 + bmin_px**2)
    bm_area = bmaj * bmin * np.pi / 4 / np.log(2)
    px_area = dx_px * dy_px
    bm_npx = bm_area / px_area
    ra_pix, dec_pix = ft.wcs2pix(fitsfile, ra, dec)
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
        maxval = np.nanmax(imdata_select)
        minval = np.nanmin(imdata_select)
        # allowing to take care of negative components
        peakval = minval if np.abs(minval) > np.abs(maxval) else maxval
        std = np.nanstd(imdata[select_err])
        inds = np.where(imdata == peakval)
        # fitting gaussian to point sources
        gauss_data = copy.deepcopy(imdata)
        gauss_data[~select] = 0            
        gauss_data = gauss_data.reshape((nxaxis, nyaxis))
        mod = models.Gaussian2D(peakval, inds[1], inds[0], bmaj_px/2, bmin_px/2, theta=bpa * np.pi/180)
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            gauss_mod = fit_p(mod, ll, mm, gauss_data)
        gauss_peak = gauss_mod.amplitude.value
        fitted_data = gauss_mod(ll, mm)
        select_err = R < 4 * bm_radius_px
        err_data = copy.deepcopy(imdata)
        err_data[~select_err] = 0
        residual = imdata - fitted_data
        gauss_int = np.sum(fitted_data) / bm_npx
        gauss_err = np.std(residual[select_err])
        peak_flux, int_flux, gauss_err, std = gauss_peak, gauss_int, gauss_err, std
    else:
        warnings.warn('WARNING: Right ascension or declination outside image field, therefore values are set to nan', Warning)
        peakval, peak_flux, int_flux, gauss_err, std = np.nan, np.nan, np.nan, np.nan, np.nan
    return {'freq': freq, 'pflux': peakval, 'gauss_pflux': peak_flux, 'gauss_tflux':int_flux, 'error': gauss_err, 'std':std}
