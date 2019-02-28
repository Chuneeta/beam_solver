import numpy as np
import healpy as hp
from pyuvdata import UVBeam

pol2ind = {'xx': 0, 'yy': 1}

def get_gaussbeam(sigma_x, sigma_y=None, mu_x=0, mu_y=0, size=31):
    """
    Generate the 2D gaussian beam using

        B_gauss = exp[-(x-mu)**2 + (y-mu)**2 / (2*sigma**2)].
    
    Parameters
    ----------
    mu : float
        Mean or centre of the normal distribution.

    sigma : float
        Standard deviation or spread of the normal distribution.

    size : int
        Dimension of the output array. Default output 31, threrefore,
	the shape of the output beam will be (31, 31) by default.
    """
    mu_x = float(mu_x)
    mu_y = float(mu_y)
    sigma_x = float(sigma_x)
    if sigma_y is None: sigma_y = sigma_x
    sigma_y = float(sigma_y)
    beam_array = np.zeros((size, size))
    bmx, bmy = np.indices(beam_array.shape)
    beam_array = np.exp(-((bmx - mu_x)**2 / (2 * sigma_x**2) + (bmy - mu_y)**2 / (2 * sigma_y**2)))
    return beam_array

def recenter(a, c):
    """Slide the (0,0) point of matrix a to a new location tuple c.  This is
    useful for making an image centered on your screen after performing an
    inverse fft of uv data."""
    s = a.shape
    c = (c[0] % s[0], c[1] % s[1])
    if np.ma.isMA(a):
        a1 = np.ma.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = np.ma.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    else:
        a1 = np.concatenate([a[c[0]:], a[:c[0]]], axis=0)
        a2 = np.concatenate([a1[:,c[1]:], a1[:,:c[1]]], axis=1)
    return a2

def get_LM(dim, center=(0,0), res=1):
        """Get the (l,m) image coordinates for an inverted UV matrix."""
        M,L = np.indices((dim, dim))
        L,M = np.where(L > dim/2, dim-L, -L), np.where(M > dim/2, M-dim, M)
        L,M = L.astype(np.float32)/dim/res, M.astype(np.float32)/dim/res
        mask = np.where(L**2 + M**2 >= 1, 1, 0)
        L,M = np.ma.array(L, mask=mask), np.ma.array(M, mask=mask)
        return recenter(L, center), recenter(M, center)

def get_top(dim, center=(0,0), res=1):
    """Return the topocentric coordinates of each pixel in the image."""
    x,y = get_LM(dim, center, res)
    z = np.sqrt(1 - x**2 - y**2)
    return x,y,z

def get_src_tracks(coord, flux, sigma_x, sigma_y=None):
    if sigma_y is None: sigma_y = sigma_x
    x, y, z = coord
    A_s = np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))
    return A_s * flux

def get_fitsbeam(filename, freq, pol='xx', nside=32):
    """
    Generate thes beam values from the beamfits for any given
    frequency using linear interpolation.

    Parameters
    ----------
    filename : string or UVBeam object
        Beamfits file or UVBeam object containing the beam values at
        different frequencies.

    freq : float
        Frequency in Hz at which the beam values needs to be interpolated.

    pol : list of str
        Polarization, can be xx, yy. Default is xx.

    nside : int
        Nside or resolution of the output healpy map.
    """

    # reading beamfits
    uvb = UVBeam()
    uvb.read_beamfits(filename)
    # peak normalization
    uvb.peak_normalize()
    data_array = uvb.data_array
    beamfreq = uvb.freq_array[0]
    beamdata = data_array[0, 0, pol2ind[pol], :, :]
    beam_array = _interp_freq(beamdata, beamfreq, freq)
    beam_array = hp.ud_grade(beam_array, nside)	

    return beam_array

def get_cstbeam(filename, beamfreq, freq, pol='xx', nside=64):
    """
    Generate beam values from the cst beam file for any given frequency 
    using linear interpolation.

    Parameters
    ----------
    filename : list of str
        Name of the CST file containing the beam model values along with the necessary
        metadata.

    beamfreq : float or list of float
        List of frequencies of the input CST beam model file(s). The frequencies should
        be given in the same order as the input filename.

    freq : float
        Frequency at which the beam values will be interpolated.

    pol : list of str
        Polarization, can be xx, xy, yx, yy. Default is xx.

    nside : int, optional
        Nside of the output healpix name. Default is 64.
    """

    # reading cst beam
    uvb = UVBeam()
    uvb.read_cst_beam(filename, beam_type='power', frequency=beamfreq,
                          telescope_name='TEST', feed_name='bob',
                          feed_version='0.1', feed_pol=pol,
                          model_name='E-field pattern - Rigging height 4.9m',
                          model_version='1.0')
    uvb.interpolation_function = 'az_za_simple'
    uvb.to_healpix(nside)
    uvb.peak_normalize()
    data_array = uvb.data_array
    beamdata = data_array[0, 0, pol2ind[pol], :, :]
    beam_array = _interp_freq(beamdata, beamfreq, freq)
    
    return beam_array

def _interp_freq(data, beamfreq, freq):
    """
    Interpolates data at any given frequency using 2 closest-point
    linear interpolation

        d = (w1*d1 + w2*d2)/ (w1 + w2)

    where d1 and d2 are the closest data points to the required frequency
    and w1 and w2 are the corresponding weight assigned to each of the data
    point.

    Parameters
    ----------
    data : np.ndarray
        2D array containg the data points with the first dimensions as
        the frequency axis.

    beamfreq: np.ndarray 
        Array containing the corresponding frequency of the data in the
        same order

    freq : float
        Frequency in Hz for which the data needs to be generated.
    """
    
    distance = np.abs(np.array(beamfreq) - freq)
    inds = np.argsort(distance)
    if freq == beamfreq[inds[0]]:
        interp_data = data[inds[0], :]
    else:
        wgt0 = (distance[inds[0]]) ** (-1)
        wgt1 = (distance[inds[1]]) ** (-1)
        interp_data = wgt0 * data[inds[0], :] + wgt1 * data[inds[1], :]
        interp_data /= (wgt0 + wgt1)

    return interp_data
