import numpy as np
from pyuvdata import UVBeam
import healpy
from collections import OrderedDict
import warnings

pol2ind = {'xx': 0, 'yy': 1}


class catBeamGauss(object):
    def __init__(self):
        """
        Object to store a simple frequency independent Gaussian beam into
        a catBeam object.
        """

    def generate_beam(self, mu, sigma, size=(31, 31)):
        """
        Generate the 2D gaussian beam using

                B_gauss = exp[-(x-mu)**2 + (y-mu)**2 / (2*sigma**2)].

        Parameters
        ----------
        mu : float
            Mean or centre of the normal distribution.

        sigma : float
            Standard deviation or spread of the normal distribution.

        size : int or tuple of ints (m,n)
            Output shape. If the the given shape is m the output shape
            will be (m,m). Default output shape is (31, 31).
        """

        # convert mu and sigma to float if they are given in integers
        if isinstance(mu, (int, np.int)):
            mu = float(mu)

        if isinstance(sigma, (int, np.int)):
            sigma = float(sigma)

        # checking dimensions of the output shape
        if isinstance(size, (int, np.int)):
            size = (size, size)

        if len(size) > 2:
            raise ValueError('Only 2D gaussian (m, n) are currently supported.')

        beam_array = np.zeros((size[0], size[1]))
        bmx, bmy = np.indices(beam_array.shape)
        beam_array = np.exp(-((bmx - mu) ** 2 + (bmy - mu) ** 2) / (2 * sigma ** 2))

        size = beam_array.shape
        return {'data': beam_array, 'size': size, 'normalization': 'gauss', 'beam_type': '2-Dimensional'}


class catBeamFits(object):
    def __init__(self):
        """
        Object to store the HERA beam values generated from the beamfits
        files for a given frequency into a catBeam object.
        """

    def generate_beam(self, filename, freq, pol=['xx']):
        """
        Generate thes beam values from the beamfits model through linear
        interpolation along the frequency axis.

        Parameters
        ----------
        filename : string or UVBeam object
            Beamfits file or UVBeam object containing the beam values at
            different frequencies.

        freq : float
            Frequency in Hz at which the beam values needs to be interpolated.

        pol : list of str
            Polarization, can be xx, xy, yx, yy. Default is xx.
        """

        # ensuring polarizations are given as a list
        if isinstance(pol, (str, np.str)):
            pol = [pol]

        if isinstance(filename, (str, np.str)):
            uvb = UVBeam()
            try:
                uvb.read_beamfits(filename)
            except ValueError:
                print ('Beam file should be in .beamfits format.')
        elif isinstance(filename, UVBeam):
            uvb = filename
        else:
            raise ValueError('Beam format is not recognized, should be either a string or pyuvdata.UVBeam object.')

        # peak normalization
        uvb.peak_normalize()
        data_array = uvb.data_array
        beamfreq = uvb.freq_array[0]

        # checking frequency input
        if not isinstance(freq, (float, int)):
            raise ValueError('Frequency value is not recognized, should be integer or float.')
        # checking frequency range
        if not (np.min(beamfreq) < freq < np.max(beamfreq)):
            raise ValueError('Specified frequency is out of the available frequency range (100e6 < freq < 200e6 Hz).')

        beam_array = OrderedDict()
        # frequency interpolation
        ind = np.where(beamfreq == freq)[0]
        for p in pol:
            if p == 'xy' or p == 'yx':
                raise NotImplementedError('Cross polarizations beams are not currently implemented.')
            beamdata = data_array[0, 0, pol2ind[p], :, :]
            if ind.size == 1:
                beam_array[p] = beamdata[ind, :]
            else:
                inds = np.argsort(beamfreq - freq)
                beam_array[p] = 0.5 * (beamdata[inds[0]] + beamdata[inds[1]])

        nside = beam_array[p].size
        return {'data': beam_array, 'size': nside, 'normalization': 'peak', 'beam_type': 'healpix'}


class catBeamCst(object):
    def __init__(self):
        """
        Object to store the beam values generated from the cst beam
        files for a given frequency into a catBeam object.
        """

    def read_cstbeam(self, filename, freq, nside=64):
        """
        Reads in beam model file and returns the beam map interpolate onto a healpix grid.

        Parameters
        ---------
        filename : str
            Name of the CST file containing the beam model values along with the necessary
            metadata.

        freq : float
            Frequency of the input CST beam model file.

        nside : int, optional
            Nside of the output healpix name. Default is 64.
        """

        uvb = UVBeam()
        uvb.read_cst_beam(filename, beam_type='power', frequency=freq,
                          telescope_name='TEST', feed_name='bob',
                          feed_version='0.1', feed_pol='xx',
                          model_name='E-field pattern - Rigging height 4.9m',
                          model_version='1.0')
        uvb.peak_normalize()
        uvb.interpolation_function = 'az_za_simple'
        uvb.to_healpix(nside)

        return uvb

    def generate_beam(self, filename, beamfreq, freq, nside=64, pol=['xx']):
        """
        Generate beam values from the cst beam file through linear
        interpolation along the frequency axis.

        Parameters
        ---------
        filename : str or UVBeam object or list of str or UVBeam object
            Name of the CST file containing the beam model values along with the necessary
            metadata.

        beamfreq : float or list of float
            List of frequencies of the input CST beam model file(s). The frequencies should
            be given in the same order as the input filename.

        freq : float
            Frequency at which the beam values will be interpolated.

        nside : int, optional
            Nside of the output healpix name. Default is 64.

        pol : list of str
            Polarization, can be xx, xy, yx, yy. Default is xx.
        """

        # ensuring filename, freq and polarizations is a list
        if isinstance(filename, (str, np.str)):
            filename = [filename]
        if isinstance(pol, (str, np.str)):
            pol = [pol]
        if not isinstance(beamfreq, list):
            beamfreq = [beamfreq]

        # size of beam files or UVBeam objects should be consistent with beamfreqs
        assert len(filename) == len(beamfreq), 'Length of beam frequencies is not consistent with beamfiles or UVBeam objects.'

        # checking input frequency
        if not isinstance(freq, (float, int)):
            raise ValueError('Frequency value is not recognized, should be integer or float.')

        bfreq_min = np.min(beamfreq)
        bfreq_max = np.max(beamfreq)
        bfreq_buffer = 5  # buffer of 5 MHz of a single frequency file is given

        beam_array = OrderedDict()
        if len(filename) == 0:
            raise ValueError('No beam files or UVBeam objects have been specified')

        elif len(filename) == 1:
            if not (bfreq_min - bfreq_buffer < freq < bfreq_max + bfreq_buffer):
                raise ValueError('Specified frequency is beyond the available frequency range ({} < freq < {} Hz).'.format(bfreq_min - bfreq_buffer, bfreq_max + bfreq_buffer))

            if freq != beamfreq:
                warnings.warn('WARNING: The specified frequency is not equal to the beam frequency, hence the output beam will at {} Hz'. format(beamfreq))

            if isinstance(filename[0], (str, np.str)):
                uvb = self.read_cstbeam(filename[0], beamfreq[0], nside=nside)
            else:
                uvb = filename[0]

            for p in pol:
                beam_array[p] = uvb.data_array[0, 0, pol2ind[p], 0, :]

        else:
            if not (bfreq_min < freq < bfreq_max):
                raise ValueError('The specified frequency is beyond the available frequency range ({} < freq < {} Hz).'.format(bfreq_min, bfreq_max))

            # finding the files corresponding to the specified frequency value
            ind = np.where(np.array(beamfreq) == freq)[0]
            if len(ind) == 1:
                uvb = self.read_cstbeam(filename[ind[0]], beamfreq[ind[0]])
                for p in pol:
                    beam_array[p] = uvb.data_array[0, 0, pol2ind[p], 0, :]
            else:
                # frequency interpolation
                inds = np.argsort(np.array(beamfreq) - freq)
                uvb1 = self.read_cstbeam(filename[inds[0]], beamfreq[inds[0]], nside)
                uvb2 = self.read_cstbeam(filename[inds[1]], beamfreq[inds[0]], nside)
                for p in pol:
                    beam_array[p] = 0.5 * (uvb1.data_array[0, 0, pol2ind[p], 0, :] + uvb2.data_array[0, 0, pol2ind[p], 0, :])

        nside = beam_array[p].size
        return {'data': beam_array, 'size': nside, 'normalization': 'peak', 'beam_type': 'cst'}
