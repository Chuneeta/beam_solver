import numpy as np
import extract as et
import coord_utils as ut
import warnings
import h5py
import os, sys
from parameter import catParam

class catData(object):
    """
    An object for storing the different source positions in celestial and horizontal coordinate system, flux values and frequencies
    """
    def __init__(self):
        # attributes of object
        # data arrays
        self.pflux_array = catParam('pflux_array', description='Peak fluxes', expected_type=float, form='(nsources, nfitsfiles/ntimes)')
        self.tflux_array = catParam('tflux_array', description='Total /integrated fluxes', expected_type=float, form='(nsources, nfitsfiles/ntimes)')
        self.pcorr_array = catParam('pcorr_array', description='Corrected peak flux', expected_type=float, form='(nsources)')
        self.tcorr_array = catParam('tcorr_array', description='Corrected total/integrated flux', expected_type=float, form='(nsources)')
        
        # meta data
        self.ras = catParam('ras', description='Right ascensions in degrees', expected_type=float, form='(nsources)')
        self.decs = catParam('decs', description='Declination in degrees', expected_type=float, form='(nsources)')
        self.azalt_array = catParam('azalts', description='Azimuth-Alitudes values in degrees', expected_type=float, form='(nsources, nfitsfiles/ntimes)')
        self.ha_array = catParam('ha_array', description='Hour angles in hours', expected_type=float, form='(nsources, nfitsfiles/ntimes)')
        self.lst_array = catParam('lst_array', description='Local Sidereal Time in hours at zenith', expected_type=float, form='(nfitsfiles/ntimes)')
        self.jd_array = catParam('jd_array', description='Julian Date', expected_type=float, form='(nfitsfiles/ntimes)')
        self.freq_array = catParam('freq_array', description='Frequency', expected_type=float, form='(nfitsfiles/ntimes)')
        self.rms_array = catParam('rms_array', description='Root mean square', expected_type=float, form='(nsources, nfitsfiles/ntimes)')
        self.beam_array = catParam('beam_array', description='Beam model values', expected_type=float, form='(nsources, nfitsfiles/ntimes)')
        self.Nfits = catParam('Nfits', description='Number of fitsfiles/times', expected_type=int)
        self.Nsrcs = catParam('Nsrc', description='Number of sources', expected_type=str)
        self.beam_type = catParam('beam_type', description='Beam type', expected_type=str)
        self.beam_size = catParam('beam_size', description='Beam size', expected_type=str)
        self.beam_normalization = catParam('beam_normalization', description='Beam normalization', expected_type=int)

    def get_data(self, key):
        """
        Slice pflux_array with a specified key in the format (ra, dec), ra and dec should be in
        degrees.

        Parameters
        ----------
        key : tuple of floats
            ra-dec pair in degrees

        Returns
        -------
        srcdict : dict
            Dictionary with peak flux, total flux, hour angle, azimuth-altitude values for 
            specified key.
        """
        
        ind = self.key_to_indices(key)

        srcdata = {}
        srcdata['pfluxs'] = self.pflux_array[ind, :]
        srcdata['tfluxs'] = self.tflux_array[ind, :]
        srcdata['pcorr'] = self.pcorr_array[ind]
        srcdata['tcorr'] = self.tcorr_array[ind]        

        return srcdata 

    def key_to_indices(self, key):
        """
        Convert key (ra-dec pair) into relevants slice arrays. The key taks the form (ra, dec) where ra and dec are in degrees.
        
        Parameters
        ----------
        key : tuple of floats
            ra-dec pair in degrees

        Returns
        -------
        indices : int
            Index of the ra-dec pair
        """

        ind_ra = np.where(self.ras == key[0])[0]
        ind_dec = np.where(self.decs == key[1])[0]        

        assert ind_ra == ind_dec, "the input key does not exists"
        ind = ind_ra    

        return ind

    def _write_mdata(self, mdata):
        """
        Write meta data information to HDF5 file
        
        Parameters
        ----------

        """
        
        mdata['ras'] = self.ras
        mdata['decs'] = self.decs
        mdata['lsts'] = self.lst_array
        mdata['jds']  = self.jd_array
        mdata['has'] = self.ha_array
        mdata['freqs'] = self.freq_array
        mdata['azalts'] = self.azalt_array
        mdata['rms'] = self.rms_array
        mdata['beam'] = self.beam_array
        mdata['Nfits'] = self.Nfits
        mdata['Nsrcs'] = self.Nsrcs
        mdata['beam_normalization'] = self.beam_normalization
        mdata['beam_type'] = self.beam_type
        mdata['beam_size'] = self.beam_size       
        
    def write_hdf5(self, filename, clobber=False):
        """
        Writes catBeam object to HDF5 file
        
        Parameters
        ----------
        filename : str
            Name of output HDF5 file

        clobber : boolean
            Option to overwrite the file if it already exists. Default is False.       
        """
        
        # checking if file exists
        if os.path.exists(filename):
            if clobber:
                print ("Overwriting existing file")
            else:   
                raise ValueError("File exists, skipping")

        with h5py.File(filename, 'w') as f:
            # write meta data
            mgp = f.create_group('Metadata')
            self._write_mdata(mgp)

            # write data to file
            dgp = f.create_group('Data')
            dgp.create_dataset('pfluxs', chunks=True, data=self.pflux_array)
            dgp.create_dataset('tfluxs', chunks=True, data=self.tflux_array)
            dgp.create_dataset('pcorrs', chunks=True, data=self.pcorr_array)
            dgp.create_dataset('tcorrs', chunks=True, data=self.tcorr_array)
                        
    def _read_mdata(self, mdata):
        """
        Reads meta data information from HDF5 files

        Parameters
        ----------
        mdata : HDF5 group
            HDF5 (h5py) data group that contains the meta data information        
        """

        self.ras = mdata['ras'].value
        self.decs = mdata['decs'].value
        self.lst_array = mdata['lsts'].value
        self.jd_array = mdata['jds'].value
        self.ha_array = mdata['has'].value
        self.freq_array = mdata['freqs'].value
        self.azalt_array = mdata['azalts'].value
        self.rms_array = mdata['rms'].value
        self.beam_array = mdata['beam'].value
        self.Nfits = mdata['Nfits'].value
        self.Nsrcs = mdata['Nsrcs'].value
        self.beam_normalization = mdata['beam_normalization'].value
        self.beam_type = mdata['beam_type'].value
        self.beam_size = mdata['beam_size'].value
    
    def read_hdf5(self, filename):
        """
        Reads in data and metadata from HDF5 file

        Parameters
        ----------
        filename : str
            Name of HDF5 (h5py) file to read in
        """
    
        if not os.path.exists(filename):
            raise IOError(filename + ' not found')

        with h5py.File(filename, 'r') as f:
            # read meta data information
            mgp = f['Metadata']
            self._read_mdata(mgp)

            # read data
            dgp = f['Data']
            self.pflux_array = dgp['pfluxs'].value
            self.tflux_array = dgp['tfluxs'].value
            self.pcorr_array = dgp['pcorrs'].value
            self.tcorr_array = dgp['tcorrs'].value         

    def check(self):
        """ 
        Run checks to make sure metadata and data arrays are properly defined and are in 
        the appropriate formats
        """
        pass        
