import numpy as np
from pyuvdata import UVBeam
import healpy
from collections import OrderedDict

poldict = {'xx':0, 'yy':1}

class catBeamGauss(object):
    def __init__(self, mu, sigma, size=None):
        """
        Object to store a simple frequency independent Gaussian beam into 
        a catBeam object.
    
        Parameters
        ----------
        mu : float 
            Mean or centre of the normal distribution.

        sigma : float
            Standard deviation or spread of the normal distribution.

        size : int or tuple of ints (m,n)
            Output shape. If the the given shape is n the output shape 
            will be (m, m). Default output shape is (31, 31). 
        """

        self. mu = mu
        self.sigma = sigma
        self.size = size

    def generate_beam(self):
        """
        Generate the 2D gaussian beam using
        
                B_gauss = exp[-(x-mu)**2 + (y-mu)**2 / (2*sigma**2)].
        """

        # convert mu and sigma to float if they are given in integers
        if isinstance(self.mu, (int, np.int)): self.mu = float(self.mu)
        if isinstance(self.sigma, (int, np.int)): self.sigma = float(self.sigma)

        # checking dimensions of the output shape
        if isinstance(self.size, int): self.size = (self.size, self.size)
        assert len(self.size) == 2, 'Only 2D gaussian (m, n) are currently supported.'

        beam_array = np.zeros((self.size[0], self.size[1]))
        bmx, bmy = np.indices(beam_array.shape)
        beam_array = np.exp(-((bmx - self.mu)**2 + (bmy - self.mu)**2)/ (2 * self.sigma**2))

        size = beam_array.shape
        return {'data':beam_array[0], 'size':size, 'normalization':'gauss', 'beam_type': '2-Dimensional'}

class catBeamFits(object):
    def __init__(self, beam, freq=None):
        """
        Object to store the HERA beam values generated from the beamfits 
        files for a given frequency into a catBeam object.

        Parameters
        ----------
        beam : string or UVBeam object 
            Beamfits file or UVBeam object containing the beam values at 
            different frequencies

        freq : float
            Frequency in Hz at which the beam values needs to be interpolated 
        """
        
        self.beam = beam
        self.freq = freq
        
    def generate_beam(self):
        """
        Generate thes beam values from the beamfits model through linear 
        interpolation along the frequency axis.
        """

        if isinstance(self.beam, (str, np.str)):
            uvb = UVBeam()
            try:
                uvb.read_beamfits(self.beam)
            except:
                raise ValueError('Beam file should be in .beamfits format.')
        elif isinstance(self.beam, UVBeam):
            uvb = self.beam
        else:
            raise ValueError('Beam format is not recognized, should be either a string or pyuvdata.UVBeam object.')
               
        # peak normalization
        uvb.peak_normalize()
        data_array = uvb.data_array
        beamfreq = uvb.freq_array[0]
        
        if self.pol=='xy' or self.pol=='yx':
            raise NotImplementedError('Cross polarizations beams are not currently implemented.')
        
        beamdata_xx = data_array[0, 0, poldict['xx'], :, :] # xx polarization
        beamdata_yy = data_array[0, 0, poldict['yy'], :, :] # yy polarization
        
        # frequency interpolation
        beam_array = OrderedDict()
        ind = np.where(beamfreq == self.freq)[0]
        if len(ind) == 1:
            beam_array['xx'] = beamdata_xx[ind, :]
            beam_array['yy'] = beamdata_yy[ind, :]	
        else:
            inds = n.argsort(beamfreq - self.freq)
            beam_array['xx'] = 0.5 * (beamdata_xx[inds[0]] + beamdata_xx[inds[1]])
            beam_array['yy'] = 0.5 * (beamdata_yy[inds[0]] + beamdata_yy[inds[1]])
        nside = len(beam_array[0])
        return {'data':beam_array, 'size':nside, 'normalization':'peak', 'beam_type': 'healpix'}

class catBeamCst(object):
    def __init__(self, beam=[], beamfreq=[], freq=None, pol='xx'):
        """
        Object to store the beam values generated from the cst beam 
        files for a given frequency into a catBeam object.

        Parameters
        ----------
        beam : list string or UVBeam object
            Beamfits file or UVBeam object containing the beam values at
            different frequencies

        beamfreqs : list of floats
            List of frequecies corresponding to the input beamfiles or UVBeam objects
        
        freq : float
            Frequency in Hz at which the beam values needs to be interpolated

        pol : string
            Polarization, can be xx, xy, yx, yy. Default is xx.
        """
        
        self.beam = beam
        self.beamfreq = beamfreq
        self.freq = freq
        self.pol = pol

    def read_cstbeam(self, beam, beamfreq):
        """
        Reads in beam model file and returns the beam map interpolate onto a healpix grid
        """
        
        uvb = UVBeam()
        uvb.read_cst_beam(beam, beam_type='power', frequency=beamfreq,
                  telescope_name='TEST', feed_name='bob',
                  feed_version='0.1', feed_pol='xx',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')
        uvb.peak_normalize()
        uvb.interpolation_function='az_za_simple'
        uvb.to_healpix()

        return uvb    

    def generate_beam(self):
        """
        Generate beam values from the cst beam file through linear
        interpolation along the frequency axis.
        """

        # size of beam files or UVBeam objects should be consistent with beamfreqs
        assert len(self.beam) == len(self.beamfreq), 'size of beam frequencies is not consistent with beamfiles or UVBeam ojects'
        
        beam_array = OrderedDict()
        if len(self.beam) == 0:
            raise ValueError('No beam files or UVBeam objects have been specified')

        elif len(self.beam) == 1:
            if isinstance(self.beam[0], (str, np.str)):
                uvb = self.read_cstbeam(self.beamfreq[0])           
            else:
                uvb = self.beam[0]
            beam_array['xx'] = uvb.data_array[0, 0, poldict['xx'], 0, :]
            beam_array['yy'] = uvb.data_array[0, 0, poldict['yy'], 0, :]

        else:
            # finding the files corresponding to the specified frequency value
            ind = np.where(np.array(self.beamfreq) == self.freq)[0]
            if len(ind) == 1:
                uvb = self.read_cstbeam(self.beam[ind[0]], self.beamfreq[ind[0]])
                beam_array['xx'] = uvb.data_array[0, 0, poldict['xx'], 0, :]
                beam_array['yy'] = uvb.data_array[0, 0, poldict['yy'], 0, :]
            else:
                # frequency interpolation
                inds = np.argsort(np.array(self.beamfreq) - self.freq)
                uvb1 = self.read_cstbeam(self.beam[inds[0]], self.beamfreq[inds[0]])
                uvb2 = self.read_cstbeam(self.beam[inds[1]], self.beamfreq[inds[0]])
                beam_array['xx'] = 0.5 * (uvb1.data_array[0, 0, poldict['xx'], 0, :] + uvb2.data_array[0, 0, poldict['xx'], 0, :])
                beam_array['yy'] = 0.5 * (uvb1.data_array[0, 0, poldict['yy'], 0, :] + uvb2.data_array[0, 0, poldict['yy'], 0, :])
        
        nside = len(beam_array)
        return {'data':beam_array, 'size':nside, 'normalization':'peak', 'beam_type': 'cst'}
