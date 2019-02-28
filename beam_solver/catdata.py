from collections import OrderedDict
from beam_solver import extract as et
from beam_solver import coord_utils as ct
from beam_solver import fits_utils as ft
import os, sys
import numpy as np
import h5py
import healpy

pol2ind = {'xx':0, 'yy': 1}

class catData(object):
    """
    Object for stroing different catalog of celestial sources
    """
    def __init__(self):
        self.data_array = None
        self.azalt_array = None
        self.ha_array = None
        self.pos_array = None
        self.err_array = None
        self.Nfits = None
        self.Nsrcs  = None
        self.Npols = None

    def get_unique(self, ras, decs, tol=5):
        """
        Selects only unique sources (right ascensions and declinations) from given celestial coordinates.
        Parameters
        ----------
        ras: list of floats
            List of right ascension values in degrees. Default list is empty.
        decs : list of floats
            List of declination values in degrees. Default list is empty
        tol : float
            Tolerance or radius in arcmin within which a source might be considered as the
            same source. Default is 5 arcmin.
         """
        if isinstance(ras, list): ras = np.array(ras)
        if not isinstance(ras, np.ndarray): 
            ras = np.array([ras])
        if isinstance(decs, list): decs = np.array(decs)
        if not isinstance(decs, np.ndarray):
            decs = np.array([decs])  
        # checking if size of ra is consistent with size of dec
        assert ras.size == decs.size, "Length of ras should be consistent with length of decs."
        n0 = ras.size
        n = n0
        unq_ras = np.array([])
        unq_decs = np.array([])
        while n > 0:
            inds = np.array([])
            ra_c = np.array([])
            dec_c = np.array([])
            # calculating the distance between adjacent coordinates (ra and dec) in arcmins
            dist = np.sqrt((ras - ras[0])**2 + (decs - decs[0])**2) * 60
            for i, d in enumerate(dist):
                if d < tol:  # checking if it is with the specified tolearance value
                    ra_c = np.append(ra_c, ras[i])
                    dec_c = np.append(dec_c, decs[i])
                    inds = np.append(inds, i)
            unq_ras = np.append(unq_ras, np.mean(ra_c))
            unq_decs = np.append(unq_decs, np.mean(dec_c))
            ras = np.delete(ras, inds)
            decs = np.delete(decs, inds)
            n = len(ras)
        print ('Found {} unique sources out of {}.'.format(len(unq_ras), n0))
        return unq_ras, unq_decs

    def _get_jds(self, fitsfiles):
        """
        Get list of julian dates for the list of fitsfiles
        Parameters
        ----------
        fitsfiles: Input list of fitsfiles
        """
        jds = []
        for fn in fitsfiles:
            fitsinfo = ft.get_fitsinfo(fn)
            hdr = fitsinfo['hdr']
            assert 'JD' in hdr.keys(), '{} does not have keyword JD'.format(fn)
            jds.append(hdr['JD'])
        return jds

    def gen_catalog(self, fitsfiles, ras, decs, flux_type='peak', return_data=False):
        """
        Extracts flux measurements at specified right ascension and declination values from the fitsfiles 
        and generates a catdata object containing the data and necessary metadata for   xx or yy 
        polarization. It can also return a dictionary containing the data and selected metadata.
        Parameters
        ---------
        fitsfiles : list of str
            List of of xx or yy fitsfiles that will be used to generate or extract the source catalog.
        ras : list of float
            List of right ascension values in degrees.
        decs : list of floats
            List of declination values in degrees.
        flux_type : str
            Type of flux density to store, can be either 'peak' or 'total'. 'peak' returns the peak
            pixel value selected from all the pixels confined within the synthesized beam. 'total' 
            returns the integrated flux density from a gaussian fit around the source.
        return_data : boolean
            If True, returns dictionary with the data values and selected metadata.
        """
        assert len(ras) == len(decs), "Right ascenscion array should be of the same size as declination array."
        # selecting unique ras and decs
        nsrcs = len(ras)
        nfits = len(fitsfiles)      
        # initializating source dict and numpy arrays
        srcdict = OrderedDict()
        pos_array = np.zeros((2, nsrcs))
        ha_array = np.zeros((nsrcs, nfits))
        err_array = np.zeros((1, nsrcs, nfits))
        data_array = np.zeros((1, nsrcs, nfits))
        azalt_array = np.zeros((2, nsrcs, nfits))
        jds = self._get_jds(fitsfiles)
        for ii, ra in enumerate(ras):
            pos_array[0, ii] = ra; pos_array[1, ii] = decs[ii]
            key = (round(ra, 2), round(decs[ii], 2))
            if not key in srcdict: srcdict[key] = {}
            for jj, fn in enumerate(fitsfiles):
                srcstats = et.get_peakflux(fn, ra, decs[ii])
                lst = ct.jd2lst(jds[jj])
                ha = ct.ralst2ha(ra * np.pi/180, lst)                  
                az, alt = ct.hadec2azalt(decs[ii] * np.pi/180., ha)
                ha_array[ii, jj] = ha
                err_array[0, ii, jj] = srcstats['error']
                data_array[0, ii, jj] = srcstats['flux']
                azalt_array[0, ii, jj] = az
                azalt_array[1, ii, jj] = alt
            # saving to dictionary
            srcdict[key]['data'] = data_array[:, ii,  :]
            srcdict[key]['error'] = err_array[:, ii, :]
            srcdict[key]['ha'] = ha_array[ii, :]
        # saving attributes to object
        self.data_array = data_array
        self.err_array = err_array
        self.ha_array = ha_array
        self.pos_array = pos_array
        self.azalt_array = azalt_array
        _sh = data_array.shape
        self.Npols = _sh[0]
        self.Nsrcs = _sh[1]
        self.Nfits = _sh[2]
        if return_data:
            return srcdict

    def write_hdf5(self, filename, clobber=False):
        """
        Writes catData object to HDF5 file (saves it on disk)

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
                print ("Overwriting existing file.")
            else:
                raise IOError("File exists, skipping.")
        with h5py.File(filename, 'w') as f:
            # write meta data
            mgp = f.create_group('Metadata')
            mgp['Nfits'] = self.Nfits
            mgp['Nsrcs'] = self.Nsrcs
            mgp['Npols'] = self.Npols
            mgp['ha_array'] = self.ha_array
            mgp['err_array'] = self.err_array
            mgp['pos_array'] = self.pos_array
            mgp['azalt_array'] = self.azalt_array
            # write data to file
            dgp = f.create_group('Data')
            dgp.create_dataset('data_array', chunks=True, data=self.data_array)

    def read_hdf5(self, filename):
        """
        Read HDF5 and loads the attributes into a catData object
        
        Parameters
        ----------
        filename : str
            Name of input HDF5 file.
        """

        with h5py.File(filename, 'r') as f:
            # read meta data information
            mgp = f['Metadata']
            self.Nfits = mgp['Nfits'].value
            self.Nsrcs = mgp['Nsrcs'].value
            self.Npols = mgp['Npols'].value
            self.ha_array = mgp['ha_array'].value
            self.err_array = mgp['err_array'].value
            self.pos_array = mgp['pos_array'].value
            self.azalt_array = mgp['azalt_array'].value
            # read data
            dgp = f['Data']
            self.data_array = dgp['data_array'].value

    def calc_catalog_flux(self, beam, pol):
        """
        Calculates corrected/ catalog flux values for all positions (ra, dec) using the measurements at different az-alt
        coordinates:
                    I = \sum I^obs(t, nu) * B(t, nu) / B(t, nu)

        where I^obs is the measurements at different az-alts (time) values and B is the beam model value at the
        given alt-az coordinate.

        Parameters
        ----------
        beam : np.ndarray
            Numpy array containing primary beam model values (refer to beam_utils.py).
        pol : str
            Polarization can be xx or yy.
        """ 
        azs = self.azalt_array[0, :, :]
        alts = self.azalt_array[1, :, :]
        flux_array = np.ndarray((self.Nsrcs), dtype=float)
        beam_array = np.ndarray((self.Nfits), dtype=float)
        for i in range(self.Nsrcs):
            for j in range(self.Nfits):
                beam_array[j] = healpy.get_interp_val(beam, np.pi/2 - (alts[i, j]), azs[i, j]) 
            if self.data_array.shape[0] == 1:
                flux_array[i] = np.nansum(self.data_array[0, i, :] * beam_array) / np.nansum(beam_array ** 2)
            else:
                flux_array[i] = np.nansum(self.data_array[pol2ind[pol], i, :] * beam_array) / np.nansum(beam_array ** 2)
        return flux_array
