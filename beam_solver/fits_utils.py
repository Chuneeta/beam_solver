#!/usr/bin/env python
import pywcs
import numpy as np
from astLib.astWCS import WCS
import pylab
import copy
from astropy.io import fits
from astropy import wcs
from astropy import modeling as mod
import scipy.stats as stats

def fitsInfo(fitsfile):
    """
    Gets info from fitsfile

    - fitsfile : Input fitsfile
    """
    # extracting image data and metadata
    data, hdr = fits.getdata(fitsfile, header=True) 
    freq = hdr ['CRVAL3']
    ra = hdr['CRVAL1']
    dra = np.abs(hdr['CDELT1'])
    crapix = hdr['CRPIX1']
    dec = hdr['CRVAL2']
    ddec = hdr['CRVAL2']
    cdecpix = hdr['CRPIX2']
    naxis1 = hdr['NAXIS1']
    naxis2 = hdr['NAXIS2']
    wcs1 = WCS(hdr, mode='pyfits')

    return {'data':data, 'hdr':hdr, 'wcs':wcs1, 'ra':ra, 'dec':dec, 'dra':dra, 'ddec':ddec, 'crapix':crapix, 'cdecpix':cdecpix, 'freq':freq, 'naxis1':naxis1, 'naxis2':naxis2}

def fitsstats(fitsfile, ra, dec, radius=None, stats=True, show=None, res=None):
    """
    Gets the statistics around a selected region centered at the specified ra and dec

    - fitsfile : input fitsfile
    - ra  : right ascension of the centre source in degrees
    - dec : declination of the centre source in degrees
    - radius : radius in pixels within which the source will be centered
    - stats: Outputs/ returns a dictionary containig the statistics of the selected region. Default is True
    - show : Displays a plot of the selected region. Default is False.
    - res  : Enable if the fitsfile is a residual file. Default is False
    """
    # extracting information from the fitsfile
    image = fitsInfo(fitsfile)
    data = image['data'][0, 0]
    WCS1 = image['wcs']
    hdr = image['hdr']
    naxis1 = image['naxis1']
    naxis2 = image['naxis2']
    freq = image['freq']

    # convert ra, dec to pixel coordinates
    rapix, decpix = WCS1.wcs2pix(ra, dec)
    rapix, decpix = int(rapix), int(decpix)
    
    # get image ra, dec pixels
    ra_axis = np.arange(0, naxis1)
    dec_axis = np.arange(0, naxis2)
    RA, DEC = np.meshgrid(ra_axis, dec_axis)

    # get radius coordinates
    R = np.sqrt((RA - rapix)**2 + (DEC - decpix)**2)

    # select pixels around desired region
    if radius is None:
        beam_width = np.sqrt(hdr['BMAJ']**2 + hdr['BMIN']**2)
        radius = int(beam_width / np.abs(hdr['CDELT1'])	)

    select = R < radius

    # calculate beam area in degrees^2
    beam_area = (hdr["BMAJ"] * hdr["BMIN"] * np.pi / 4 / np.log(2))
    # calculate pixel area in degrees ^2
    pixel_area = np.abs(hdr["CDELT1"] * hdr["CDELT2"])
    Npix_beam = beam_area / pixel_area
    rms = np.sqrt(np.mean(data[~select]**2))
    peak_err = rms / np.sqrt(Npix_beam / 2.0)

    data_copy = copy.deepcopy(data)
    data_copy[~select] = np.nan 
    
    # plotting the region
    if show:
        # getting coordinate system
        my_wcs = wcs.WCS(hdr, naxis=[wcs.WCSSUB_CELESTIAL])
        ax = pylab.subplot(111, projection = my_wcs)
        im = ax.imshow(data_copy, origin='lower', aspect='auto', cmap='gray')
        pylab.grid(lw=1,color='black')
        pylab.colorbar(im)
        pylab.show() 

    if stats:
        mean = np.nanmean(data_copy)
        total = np.nansum(data_copy)
        std = np.nanstd(data_copy)
        maxval = np.nanmax(data_copy)
        minval = np.nanmin(data_copy)
        if res:
            peakval = minval if np.abs(minval) > maxval else maxval
        else:
            peakval = maxval
        peak_flux = peakval # peak flux
        npix = len(data_copy[~np.isnan(data_copy)]) 
        int_flux = total / npix # integrated flux

        # returns dictionary containing the stats
        return {'peak':peakval, 'rms':rms, 'peak_err':peak_err, 'mean':mean, 'total':total, 'freq':freq, 'std':std, 'int_flux':int_flux }

def gaussfit(imfile, ra, dec, radius=1, gaussfit_mult=1.0, rms_max_r=None, rms_min_r=None, **kwargs):

    # open fits file
    hdu = fits.open(imfile)

    # get header and data
    head = hdu[0].header
    data = hdu[0].data.squeeze()

    # determine if freq precedes stokes in header
    if head['CTYPE3'] == 'FREQ':
        freq_ax = 3
        stok_ax = 4
    else:
        freq_ax = 4
        stok_ax = 3

    # get axes info
    npix1 = head["NAXIS1"]
    npix2 = head["NAXIS2"]
    nstok = head["NAXIS{}".format(stok_ax)]
    nfreq = head["NAXIS{}".format(freq_ax)]

    # calculate beam area in degrees^2
    beam_area = (head["BMAJ"] * head["BMIN"] * np.pi / 4 / np.log(2))

    # calculate pixel area in degrees ^2
    pixel_area = np.abs(head["CDELT1"] * head["CDELT2"])
    Npix_beam = beam_area / pixel_area
     # get ra dec coordiantes
    ra_axis = np.linspace(head["CRVAL1"]-head["CDELT1"]*head["NAXIS1"]/2, head["CRVAL1"]+head["CDELT1"]*head["NAXIS1"]/2, head["NAXIS1"])
    dec_axis = np.linspace(head["CRVAL2"]-head["CDELT2"]*head["NAXIS2"]/2, head["CRVAL2"]+head["CDELT2"]*head["NAXIS2"]/2, head["NAXIS2"])
    RA, DEC = np.meshgrid(ra_axis, dec_axis)

    # get radius coordinates
    R = np.sqrt((RA - ra)**2 + (DEC - dec)**2)

    # select pixels
    select = R < radius

    # get peak brightness within pixel radius
    peak = np.max(data[select])

    # get rms outside of pixel radius
    if rms_max_r is not None and rms_max_r is not None:
        rms_select = (R < rms_max_r) & (R > rms_min_r)
        rms = np.sqrt(np.mean(data[select]**2))
    else:
        rms = np.sqrt(np.mean(data[~select]**2))

    # get peak error
    peak_err = rms / np.sqrt(Npix_beam / 2.0)

    # get frequency of image
    freq = head["CRVAL3"]

    ## fit a 2D gaussian and get integrated and peak flux statistics ##
    # recenter R array by peak flux point and get thata T array
    peak_ind = np.argmax(data[select])
    peak_ra = RA[select][peak_ind]
    peak_dec = DEC[select][peak_ind]
    X = (RA - peak_ra)
    Y = (DEC - peak_dec)
    R = np.sqrt(X**2 + Y**2)
    X[np.where(np.isclose(X, 0.0))] = 1e-5
    T = np.arctan(Y / X)

    # use synthesized beam as data mask
    ecc = head["BMAJ"] / head["BMIN"]
    beam_theta = head["BPA"] * np.pi / 180 + np.pi/2
    EMAJ = R * np.sqrt(np.cos(T+beam_theta)**2 + ecc**2 * np.sin(T+beam_theta)**2)
    fit_mask = EMAJ < (head["BMAJ"] / 2 * gaussfit_mult)
    masked_data = data.copy()
    masked_data[~fit_mask] = 0.0

    # fit 2d gaussian
    gauss_init = mod.functional_models.Gaussian2D(peak, ra, dec, x_stddev=head["BMAJ"]/2, y_stddev=head["BMIN"]/2) 
    fitter = mod.fitting.LevMarLSQFitter()
    gauss_fit = fitter(gauss_init, RA[fit_mask], DEC[fit_mask], data[fit_mask])

    # get gaussian fit properties
    peak_gauss_flux = gauss_fit.amplitude.value
    P = np.array([X, Y]).T
    beam_theta -= np.pi/2
    Prot = P.dot(np.array([[np.cos(beam_theta), -np.sin(beam_theta)], [np.sin(beam_theta), np.cos(beam_theta)]]))
    gauss_cov = np.array([[gauss_fit.x_stddev.value**2, 0], [0, gauss_fit.y_stddev.value**2]])
    try:
        model_gauss = stats.multivariate_normal.pdf(Prot, mean=np.array([0,0]), cov=gauss_cov)
        model_gauss *= gauss_fit.amplitude.value / model_gauss.max()
        int_gauss_flux = np.nansum(model_gauss.ravel()) / Npix_beam
    except:
        int_gauss_flux = 0

    return {'peak':peak, 'peak_err':peak_err, 'rms':rms, 'peak_gauss_flux':peak_gauss_flux, 'int_gauss_flux':int_gauss_flux, 'freq':freq}
