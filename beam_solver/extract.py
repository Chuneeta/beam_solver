import numpy as np
from beam_solver import fits_utils as ft
from beam_solver.data import DATA_PATH
import warnings

def get_peakflux(fitsfile, ra, dec, radius=None, negative=False):
    """
    Returns the statistics obtained from the desired or selected region
    fitsfile: Input fitsfile
    ra : Right ascension in degrees
    dec : Declination in degrees
    radius: Radius or tolerance to select the desired region
    negative: If True, returns the maximum value irrespective of the
    ngative sign.
    """
    fitsinfo = ft.get_fitsinfo(fitsfile)
    imdata = fitsinfo['data']
    hdr = fitsinfo['hdr']
    freq = fitsinfo['freq']
    nxaxis = fitsinfo['nxaxis']
    nyaxis = fitsinfo['nyaxis']
    bmaj = hdr['BMAJ']
    bmin = hdr['BMIN']
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
        peakval = np.nanmax(imdata_select)
        minval = np.nanmin(imdata_select)
        if negative:
            if np.abs(peakval) < np.abs(minval):
                peakval = minval
        std = np.nanstd(imdata_select)
        rms = np.sqrt(np.nanmean(imdata[~select])**2) # calculated rms outside the selected region
        peak_err = rms / np.sqrt(bm_npx / 2.0)
        flux, err = peakval, peak_err
    else:
        warnings.warn('WARNING: Right ascension or declination outside image field, therefore values are set to nan', Warning)
        flux, err = np.nan, np.nan
    
    return {'freq': freq, 'flux': flux, 'error': err}
