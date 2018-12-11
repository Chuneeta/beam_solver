import numpy as np
from pyuvdata import UVBeam
import healpy

pol2ind = {'xx': 0, 'yy': 1}

def get_gaussbeam(mu, sigma, size=31):
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
    
    beam_array = np.zeros((size, size))
    bmx, bmy = np.indices(beam_array.shape)
    beam_array = np.exp(-((bmx - float(mu)) ** 2 + (bmy - float(mu)) ** 2) / (2 * float(sigma) ** 2))

    return beam_array

def get_fitsbeam(filename, freq, pol='xx'):
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
    uvb.peak_normalize()
    uvb.interpolation_function = 'az_za_simple'
    uvb.to_healpix(nside)
    uvb.peak_normalize()
    data_array = uvb.data_array
    beamdata = data_array[0, 0, pol2ind[pol], :, :]
    beam_array = _interp_freq(beamdata, beamfreq, freq)

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
