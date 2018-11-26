import numpy as np
import extract as et
import coord_utils as ut
import warnings
import h5py
import os, sys
from parameter import catParam
import healpy
import fnmatch

pol2ind = {'xx': 0, 'yy': 1}


class catData(object):
    """
    An object for storing the different source positions in celestial and horizontal coordinate system, flux values and frequencies
    """
    def __init__(self):
        # attributes of object
        # data arrays
        self._pflux_array = catParam('pflux_array', description='Peak fluxes', expected_type=np.float64, form='(nsources, nfitsfiles/ntimes)')
        self._tflux_array = catParam('tflux_array', description='Total /integrated fluxes', expected_type=np.float64, form='(nsources, nfitsfiles/ntimes)')
        self._pcorr_array = catParam('pcorr_array', description='Corrected peak flux', expected_type=np.float64, form='(nsources)')
        self._tcorr_array = catParam('tcorr_array', description='Corrected total/integrated flux', expected_type=np.float64, form='(nsources)')
        
        # meta data
        self._ras = catParam('ras', description='Right ascensions in degrees', expected_type=np.float64, form='(nsources)')
        self._decs = catParam('decs', description='Declination in degrees', expected_type=np.float64, form='(nsources)')
        self._azalt_array = catParam('azalt_array', description='Azimuth-Alitudes values in degrees', expected_type=np.float64, form='(nsources, nfitsfiles/ntimes)')
        self._ha_array = catParam('ha_array', description='Hour angles in hours', expected_type=np.float64, form='(nsources, nfitsfiles/ntimes)')
        self._lst_array = catParam('lst_array', description='Local Sidereal Time in hours at zenith', expected_type=np.float64, form='(nfitsfiles/ntimes)')
        self._jd_array = catParam('jd_array', description='Julian Date', expected_type=np.float64, form='(nfitsfiles/ntimes)')
        self._freq_array = catParam('freq_array', description='Frequency', expected_type=np.float64, form='(nfitsfiles/ntimes)')
        self._rms_array = catParam('rms_array', description='Root mean square', expected_type=np.float64, form='(nsources, nfitsfiles/ntimes)')
        self._beam_array = catParam('beam_array', description='Beam model values', expected_type=np.float64, form='(nsources, nfitsfiles/ntimes)')
        self._Nfits = catParam('Nfits', description='Number of fitsfiles/times', expected_type=int)
        self._Nsrcs = catParam('Nsrcs', description='Number of sources', expected_type=int)
        self._Npols = catParam('Npols', description='Number of polarizations', expected_type=int)
        self._beam_type = catParam('beam_type', description='Beam type', expected_type=str)
        self._beam_size = catParam('beam_size', description='Beam size', expected_type=int)
        self._beam_normalization = catParam('beam_normalization', description='Beam normalization', expected_type=int)
        
        self._pcorr_array = None
        self._tcorr_array = None
        self._beam_array = None
        self._beam_normalization = None
        self._beam_type = None 
        self._beam_size = None

        self._all_parameters = sorted(map(lambda p: p[1:],
                                    fnmatch.filter(self.__dict__.keys(), '_*')))

        self._req_parameters = ['ras', 'decs', 'pflux_array', 'tflux_array', 'azalt_array',
                               'ha_array', 'lst_array', 'js_array', 'rms_array', 'freq_array',
                               'Nfits', 'Nsrcs', 'Npols']

        self._opt_parameters = ['pcorr_array', 'tcorr_array', 'beam_array', 'beam_size',
                               'beam_type', 'beam_normalization']
 
        self._ndarrays = ['pflux_array', 'tflux_array', 'azalt_array',
                             'ha_array', 'lst_array', 'js_array', 'rms_array', 'freq_array',
                             'pcorr_array', 'tcorr_array', 'beam_array']

        self._extras = ['Nsrcs', 'Nfits', 'beam_normalization', 'beam_type', 'beam_size']


    def get_pflux(self, key):
        """
        Slice pflux_array for the specified key. The key takes the form ((ra, dec), pol) where ra and dec are in degrees and polation can be xx or yy.

        Parameters
        ----------
        key : tuple of (ra,dec) and pol
            ((ra,dec), pol). (ra, dec) pair in degrees and polarizations can be xx or yy.

        Returns
        -------
        srcdict : dict
            Array with the peak flux values for all alt-az for the specified key.
        """
        
        ind = self._key2ind(key)
        return self.pflux_array[ind[0], ind[1], :]

    def get_tflux(self, key):
        """
        Slice tflux_array for the specified key. The key takes the form ((ra, dec), pol) where ra and dec are in degrees and polation can be xx or yy.       
 
        Parameters
        ----------
        key : tuple of (ra,dec) and pol
            ((ra,dec), pol). (ra, dec) pair in degrees and polarizations can be xx or yy.

        Returns
        -------
        srcdict : dict
            Array with the peak flux values for all alt-az for the specified key.
        """

        ind = self._key2ind(key)
        return self.tflux_array[ind[0], ind[1], :]

    def get_pcorr(self, key):
        """
        Slice pcorr_array for the specified key. The key takes the form ((ra, dec), pol) where ra and dec are in degrees and polation can be xx or yy.

        Parameters
        ----------
        key : tuple of (ra,dec) and pol
            ((ra,dec), pol). (ra, dec) pair in degrees and polarizations can be xx or yy.

        Returns
        -------
        srcdict : dict
            Corrected peak flux value corresponding to the specified key.
        """   

        if not self.pcorr_array is None:
            raise ValueError('pcorr_array is a NoneType object.')

        ind = self._key2ind(key)
        return self.pcorr_array[ind[0], ind[1]]

    def get_tcorr(self, key):
        """
        Slice tcorr_array for the specified key. The key takes the form ((ra, dec), pol) where ra and dec are in degrees and polation can be xx or yy.

        Parameters
        ----------
        key : tuple of (ra,dec) and pol
            ((ra,dec), pol). (ra, dec) pair in degrees and polarizations can be xx or yy.

        Returns
        -------
        srcdict : dict
            Corrected integrated/total flux value to the specified key.
        """

        if not self.tcorr_array is None:
            raise ValueError('tcorr_array is a NoneType object.')

        ind = self._key2ind(key)
        return self.tcorr_array[ind[0], ind[1]]

    def get_azalt(self, ra, dec):
        """
        Slice azalt_array for the corresponding (ra, dec) pair. 

        Parameters
        ----------
        ra : float
            Right ascension in degrees.

        dec : float
            Declination in degrees.

        Returns
        -------
            Array with the azalt values for the specified (ra, dec) pair.
        """

        ind = self.radec2ind(ra, dec)
        return self.azalt_array[ind, :]    

    def get_ha(self, ra, dec):
        """
        Slice ha_array for the corresponding (ra, dec) pair.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.

        dec : float
            Declination in degrees.

        Returns
        -------
            Array with the hour angle values for the specified (ra, dec) pair.
        """

        ind = self.radec2ind(ra, dec)
        return self.ha_array[ind, :]

    def radec2ind(self, ra, dec):
        """
        Returns the array index corresponding to the releavant (ra, dec) pair.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        
        dec : float
            Declination in degrees.

        Returns
        -------
            indices : int
            Indices of the (ra,dec) pair and polarization.    
        """

        # checking of ra dec are proper
        self._check_ra(ra)
        self._check_dec(dec)

        # finding index for (ra, dec) pair
        ind_ra = np.where(self.ras == ra)[0]
        ind_dec = np.where(self.decs == dec)[0]

        if ind_ra != ind_dec:
            raise KeyError('the input (ra, dec) pair does not exists.')
        ind = ind_ra[0]

        return ind

    def _key2ind(self, key):
        """
        Convert key ((ra,dec), pol) into relevants slice arrays. The key takes the form ((ra, dec), pol) where ra and dec are in degrees and polation can be xx or yy.
        
        Parameters
        ----------
        key : tuple of (ra,dec) and pol 
            (radec) pair should in degrees and polarization can be xx or yy.

        Returns
        -------
        indices : int
            Indices of the (ra,dec) pair and polarization
        """

        # checking if key is a tuple
        if not isinstance(key, tuple):
            if isinstance(key, list):
                key = tuple(key)
            else:
                raise ValueError('The specified key should be a tuple.')

        if not isinstance(key[0], (tuple, list)):
            raise ValueError('The specified key should be a tuple.')

        # checking if the values of the key
        self._check_ra(key[0][0])
        self._check_dec(key[0][1])

        if not isinstance(key[1], (str, np.str)):
            raise ValueError('polarization should be strings (xx or yy).')

        # finding index for (ra, dec) pair
        ind1 = self.radec2ind(key[0][0], key[0][1])       

        # finding index for polarization
        pol = key[1].lower()

        if pol not in pol2ind.keys():
            raise ValueError('polarization {} could not recongnized. Supports on xx and yy polarizations.'.format(pol))
        ind2 = pol2ind[pol]

        return (ind1, ind2)

    def _check_ra(self, ra):
        """
        Checks if right ascension values are within the proper range or exists in the source catalog.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        """
        
        if not isinstance(ra, (int, np.int, float, np.float)):
            raise ValueError('Right ascension should be float or integers.')

        # checking right ascension and declination ranges
        if not ra in self.ras:
            raise KeyError('The specified right ascension value could be found.')

    def _check_dec(self, dec):
        """
        Checks if right declinations values are within the proper range or exists in the source catalog.

        Parameters
        ----------
        dec : float
            Declination in degrees.
        """

        if not isinstance(dec, (int, np.int, float, np.float)):
            raise ValueError('Declination should be float or integers.')

        # checking right ascension and declination ranges
        if not dec in self.decs:
            raise KeyError('The specified declination value could be found.')

    def _write_mdata(self, mdata):
        """
        Write meta data information to HDF5 file
        
        Parameters
        ----------
        mdata : HDF5 group
            HDF5 (h5py) data group that will contain the meta data information.
        """
    
        mdata['ras'] = self.ras
        mdata['decs'] = self.decs
        mdata['lst_array'] = self.lst_array
        mdata['jd_array']  = self.jd_array
        mdata['ha_array'] = self.ha_array
        mdata['freq_array'] = self.freq_array
        mdata['azalt_array'] = self.azalt_array
        mdata['rms_array'] = self.rms_array
        mdata['beam_array'] = self.beam_array
        mdata['Nfits'] = self.Nfits
        mdata['Nsrcs'] = self.Nsrcs
        mdata['Npols'] = self.Npols
        mdata['beam_size'] = self.beam_size
        mdata['beam_type'] = self.beam_type
        mdata['beam_normalization'] = self.beam_normalization

    def write_hdf5(self, filename, run_check=True, clobber=False):
        """
        Writes catBeam object to HDF5 file
        
        Parameters
        ----------
        filename : str
            Name of output HDF5 file

        run_check: boolean
            Checks if all the attributes of the object are proper (array shapes and data types).
            Default is True.

        clobber : boolean
            Option to overwrite the file if it already exists. Default is False.       
        """
        
        # checking if file exists
        if os.path.exists(filename):
            if clobber:
                print ("Overwriting existing file.")
            else:   
                raise IOError("File exists, skipping.")

        # checks the object attributes
        if run_check:
            self.check

        with h5py.File(filename, 'w') as f:
            # write meta data
            mgp = f.create_group('Metadata')
            self._write_mdata(mgp)

            # write data to file
            dgp = f.create_group('Data')
            dgp.create_dataset('pflux_array', chunks=True, data=self.pflux_array)
            dgp.create_dataset('tflux_array', chunks=True, data=self.tflux_array)
            dgp.create_dataset('pcorr_array', chunks=True, data=self.pcorr_array)
            dgp.create_dataset('tcorr_array', chunks=True, data=self.tcorr_array) 

    def _read_mdata(self, mdata, run_check=True):
        """
        Reads meta data information from HDF5 files

        Parameters
        ----------
        mdata : HDF5 group
            HDF5 (h5py) data group that contains the meta data information.        
        """
  
        self.ras = mdata['ras'].value
        self.decs = mdata['decs'].value
        self.lst_array = mdata['lst_array'].value
        self.jd_array = mdata['jd_array'].value
        self.ha_array = mdata['ha_array'].value
        self.freq_array = mdata['freq_array'].value
        self.azalt_array = mdata['azalt_array'].value
        self.rms_array = mdata['rms_array'].value
        self.beam_array = mdata['beam_array'].value
        self.Nfits = mdata['Nfits'].value
        self.Nsrcs = mdata['Nsrcs'].value
        self.Npols = mdata['Npols'].value
        self.beam_type = mdata['beam_type'].value    
        self.beam_size = mdata['beam_size'].value
        self.beam_normalization = mdata['beam_normalization'].value

    def read_hdf5(self, filename, run_check=True):
        """
        Reads in data and metadata from HDF5 file

        Parameters
        ----------
        filename : str
            Name of HDF5 (h5py) file to read in.

        run_check: boolean
            Checks if all the attributes of the object are proper (array shapes and data types).
            Default is True.
        """
    
        if not os.path.exists(filename):
            raise IOError(filename + ' not found')

        with h5py.File(filename, 'r') as f:
            # read meta data information
            mgp = f['Metadata']
            self._read_mdata(mgp)

            # read data
            dgp = f['Data']
            self.pflux_array = dgp['pflux_array'].value
            self.tflux_array = dgp['tflux_array'].value
            self.pcorr_array = dgp['pcorr_array'].value
            self.tcorr_array = dgp['tcorr_array'].value
 
        # checks the object attributes
        if run_check:
            self.check()

    def calc_corrflux(self, beam=None, pol=['xx'], run_check=True):
        """
        Calculates corrected flux values for all positions (ra, dec) using the measurements at different az-alt
        coordinates:
                    I = \sum I^obs(t, nu) * B(t, nu) / B(t, nu)
        
        where I^obs is the measurements at different az-alts (time) values and B is the beam model value at the
        given alt-az coordinate.

        Parameters
        ----------
        beam : catBeam object or dict
            catBeam object or dict containing the primary beam model values.

        pol : list of str
            List of polarizations (xx, yy, xy, yx).

        run_check: boolean
            Checks if all the attributes of the object are proper (array shapes and data types).
            Default is True.
        """
        
        beampols = beam['data'].keys()	

        azs = self.azalt_array[0, :, :]
        alts = self.azalt_array[1, :, :]
        _sh = alts.shape
        nsrc = _sh[0]
        nfits = _sh[1]
        beam_array = np.zeros((len(pol), nsrc, nfits))
        pcorr_array = np.zeros((len(pol), nsrc))
        tcorr_array = np.zeros((len(pol), nsrc))

        for i in range(nsrc):
            for j in range(nfits):
                for p in pol:
                    if not p in beampols:
                        raise ValueError('Could not find beam value for {} polarization.'.format(p))	
                    
                    beam_array[pol2ind[p], i, j] = healpy.get_interp_val(beam['data'][p], np.pi/2 - alts[i, j] * np.pi/180., azs[i, j] * np.pi/180.)

            pcorr_array[pol2ind[p], i] = np.nansum(self.pflux_array[pol2ind[p], i, :] * beam_array[pol2ind[p], i, :]) / np.nansum(beam_array[pol2ind[p], i, :] ** 2)
            tcorr_array[pol2ind[p], i] = np.nansum(self.tflux_array[pol2ind[p], i, :] * beam_array[pol2ind[p], i, :]) / np.nansum(beam_array[pol2ind[p], i, :] ** 2)
      
        # fill in data into catdata object
        self.beam_array = beam_array
        self.pcorr_array = pcorr_array
        self.tcorr_array = tcorr_array

        # fill in metadata
        self.beam_type = beam['beam_type']
        self.beam_size = beam['size']
        self.beam_normalization = beam['normalization']

        #checking the attribute of catData object
        #if run_check:
        #    self.check()    

    def check(self):
        """ 
        Run checks to make sure metadata and data arrays are properly defined and are in 
        the appropriate formats
        
        """
        
        for p in self._all_parameters:
            # checking if the attributes exists and their types
            # required parameters
            if p in self._req_parameters:
                attr = getattr(self, '_' + p)
                assert hasattr(self, p), 'required parameter {} does not exist'.format(p)
                # numpy arrays
                if p in self._ndarrays:
                    assert isinstance(getattr(self, p), np.ndarray), 'attribute {} needs to be an ndarray'.format(p)
                    if issubclass(getattr(self, p).dtype.type, attr.expected_type):
                        pass
                    else:
                        # try to cast the attribte into its dtype
                        try:
                            setattr(self, p, getattr(self, p).astype(attr.expected_type))
                        except:
                            raise ValueError('attribute {} does not have expected data type {}'.format(p, attr.expected_type))
                # extras metadata
                if p in self._extras:
                    if not isinstance(getattr(self, p), attr.expected_type):
                        try:
                            setattr(self, p, attr.expected_type(getattr(self, p)))
                        except:
                            raise ValueError('attribute {} does no have expected data type {}'.format(p, attr.expected_type))
         
        # check for optional attributes
        if not self.pcorr_array is None:
            assert isinstance(self.pcorr_array, np.ndarray), 'attribute tcorr_array needs to be an ndarray'
        if not self.tcorr_array is None:
            assert isinstance(self.tcorr_array, np.ndarray), 'attribute tcorr_array needs to be an ndarray'
        if not self.beam_size is None:
            assert isinstance(self.beam_size, (int, np.int)), 'attribute beam_size needs to be an interger.'
        if not self.beam_normalization is None:
            assert isinstance(self.beam_normalization, (str, np.str)), 'attribute beam_normalization needs to be an string.'
        
        if not isinstance(self.pcorr_array.dtype, (float, np.float)):
            try:
                self.pcorr_array.astype(float)
            except:
                raise ValueError('attribute pcorr_array does not have expected data type float.')
        
	if not isinstance(self.tcorr_array.dtype, (float, np.float)):
            try:
                self.tcorr_array.astype(float)
            except:
                raise ValueError('attribute pcorr_array does not have expected data type.float.')
        if not self.beam_type is None:
            if not self.beam_type in ['healpix', 'cst', 'None']:
                raise ValueError('Beam type should be either healpix or CST.')

        # checking shapes of ndarrays
        nfits = self.Nfits
        nsrcs = self.Nsrcs
        npols = self.Npols
    
        # right ascension and declinations
        assert len(self.ras) == nsrcs, 'Length of ras should be same as Nsrcs.'
        assert len(self.decs) == nsrcs, 'Length of decs should be same as Nsrcs.'
            
        # flux array
        _sh = self.pflux_array.shape
        assert _sh == (npols, nsrcs, nfits), 'pflux_array should be of shape ({},{})'.format(nsrcs, nfits)
        
        _sh = self.tflux_array.shape
        assert _sh == (npols, nsrcs, nfits), 'tflux_array should be of shape ({},{})'.format(nsrcs, nfits)

        # azalt array
        _sh = self.azalt_array.shape
        assert _sh == (2, nsrcs, nfits), 'azalt array should be of shape ({},{})'.format(2, nsrcs, nfits)

        # rms array
        _sh = self.rms_array.shape
        assert _sh == (npols, nsrcs, nfits), 'rms_array should be of shape ({},{})'.format(nsrcs, nfits)
        
        # ha array
        _sh = self.ha_array.shape
        assert _sh == (nsrcs, nfits), 'ha_array should be of shape ({},{})'.format(nsrcs, nfits)
        
        # lst array
        _sh = self.lst_array.shape
        assert _sh == (nsrcs, nfits), 'lst_array should be of shape ({},{})'.format(nsrcs, nfits)
        
        # jd array
        _sh = self.jd_array.shape
        assert _sh == (nfits,), 'jd_array should be of shape ({})'.format(nfits)

        # freq array
        _sh = self.freq_array.shape
        print _sh, nfits
	assert _sh == (nfits,), 'freq_array should be of shape ({})'.format(nfits)

	# corrected flux array
        if not self.pcorr_array is None:
            _sh = self.pcorr_array.shape
            assert _sh == (npols, nsrcs), 'pcorr_array should be of shape ({},{})'.format(npols, nsrcs)
        if not self.tcorr_array is None:
            _sh = self.tcorr_array.shape
            assert _sh == (npols, nsrcs), 'tcorr_array should be of shape ({},{})'.format(npols, nsrcs)
        # beam array
        if not self.beam_array is None:
            _sh = self.beam_array.shape
            assert _sh == (npols, nsrcs, nfits), 'beam_array should be of shape ({},{})'.format(nsrcs, nfits)
