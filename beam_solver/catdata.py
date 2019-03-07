from collections import OrderedDict
from beam_solver import extract as et
from beam_solver import coord_utils as ct
from beam_solver import fits_utils as ft
import scipy
from scipy import interpolate
import os, sys
import numpy as np
import h5py
import healpy
import copy

pol2ind = {'xx':0, 'yy':1}

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
        self.pols = None

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

    def _get_lstha(self, jd, ra):
        """
        Returns local sidereal time and hour angle in radians
        jd : float
            Julian date representing the observating time
        ra : float
            Right ascension in radians
        """
        lst = ct.jd2lst(jd)
        ha = ct.ralst2ha(ra * np.pi/180, lst)
        return lst, ha

    def _get_azalt(self, dec, ha):
        """
        Return az alt coordinates in radians
        Parameters
        ---------
        dec : float
            Declination in degrees.
        ha : float 
            Hour angle in degrees.       
        """
        az, alt = ct.hadec2azalt(dec * np.pi/180., ha)
        return az, alt

    def _generate_srcdict(self, ras, decs, fitsfiles):
        """
        Extracts flux measurements at specified right ascension and declination values from the fitsfiles
        and return dictionary containing the data and necessary metadata for single polarization.
        polarization. 
        Parameters
        ---------
        fitsfiles : list of str
            List of of xx or yy fitsfiles that will be used to extract the flux values.
        ras : list of float
            List of right ascension values in degrees.
        decs : list of floats
            List of declination values in degrees.
        """
        # selecting unique ras and decs
        nsrcs = len(ras)
        nfits = len(fitsfiles)
        srcdict = OrderedDict()
        ha_array = np.zeros((nsrcs, nfits))
        err_array = np.zeros((nsrcs, nfits))
        data_array = np.zeros((nsrcs, nfits))
        azalt_array = np.zeros((2, nsrcs, nfits))
        jds = self._get_jds(fitsfiles)
        for ii, ra in enumerate(ras):
            key = (round(ra, 2), round(decs[ii], 2))
            if not key in srcdict: srcdict[key] = {}
            for jj, fn in enumerate(fitsfiles):
                srcstats = et.get_peakflux(fn, ra, decs[ii])
                lst, ha = self._get_lstha(jds[jj], ra)
                az, alt = self._get_azalt(decs[ii], ha)
                ha_array[ii, jj] = ha
                err_array[ii, jj] = srcstats['error']
                data_array[ii, jj] = srcstats['flux']
                azalt_array[0, ii, jj] = az
                azalt_array[1, ii, jj] = alt
                # saving to dictionary
            srcdict[key]['data'] = data_array[ii,  :]
            srcdict[key]['error'] = err_array[ii, :]
            srcdict[key]['ha'] = ha_array[ii, :]
            srcdict[key]['azalt'] = azalt_array[:, ii, :]
        return srcdict

    def _srcdict_catdata(self, srcdict):
        """
        Created catData object using dictionary containing the data values and required metadata.
        Parameters
        ----------
        srcdict : dict
            Dictionary containing the data and metadata about the astronomical sources.
        """ 
        # saving attributes to object
        keys = srcdict.keys()
        self.pos_array = keys
        _sh1 = len(keys)
        _sh = srcdict[keys[0]]['data'].shape
        if len(_sh) == 1:
            _sh0 = 1; _sh2 = _sh[0]
        else:
            _sh0 = _sh[0]; _sh2 = _sh[1]  
        self.data_array = np.zeros((_sh0, _sh1, _sh2)) 
        self.err_array = np.zeros((_sh0, _sh1, _sh2))
        self.ha_array = np.zeros((_sh1, _sh2))
        self.azalt_array= np.zeros((2, _sh1, _sh2))
        for ii, key in enumerate(keys):
            self.data_array[:, ii, :] = srcdict[key]['data']
            self.err_array[:, ii, :] = srcdict[key]['error']
            self.ha_array[ii, :] = srcdict[key]['ha']
            self.azalt_array[:, ii, :] = srcdict[key]['azalt']
        self.Nsrcs = _sh1
        self.Nfits = _sh2

    def _combine_srcdict(self, srcdict_xx, srcdict_yy):
        """
        Combines data from two dictionaries into a single one
        srcdict_xx: dict
            Dictionary containing data and metadata for xx polarization.
        srcdict_yy: dict
            Dictionary containing data and metadata for yy polarization.
        """
        keys_xx = srcdict_xx.keys()
        keys_yy = srcdict_yy.keys()
        srcdict = copy.deepcopy(srcdict_xx)
        assert keys_xx == keys_yy, "both dictionary should have the same keywords."
        for key in keys_xx:
            for skey in ['data', 'error']:
                srcdict[key][skey] = np.array([srcdict_xx[key][skey], srcdict_yy[key][skey]])
        return srcdict

    def gen_catalog(self, ras, decs, fitsfiles_xx, fitsfiles_yy=None, pols='xx', return_data=False):
        """
        Extracts flux measurements at specified right ascension and declination values from the fitsfiles
        and generates a catdata object containing the data and necessary metadata for xx or yy or both
        polarization. It can also return a dictionary containing the data and selected metadata.
        Parameters
        ---------
        ras : list of float
            List of right ascension values in degrees.
        decs : list of floats
            List of declination values in degrees.
        fitsfiles_xx : list of str
            List of of xx fitsfiles that will be used to generate or extract the source catalog.
        fitsfiles_yy : list of str
            List of of yy fitsfiles that will be used to generate or extract the source catalog.
        pol : str ot list of str
            Polizations can be xx or yy or both.
        return_data : boolean
            If True, returns dictionary with the data values and selected metadata.
        """
        assert len(ras) == len(decs), "Right ascenscion array should be of the same size as declination array."
        if not isinstance(pols, list): pols = [pols]
        npols = len(pols)
        if npols == 1:
            fitsfiles = fitsfiles_xx if pols[0] == 'xx' else fitsfiles_yy 
            srcdict = self._generate_srcdict(ras, decs, fitsfiles_xx) 
        else:
            srcdict_xx = self._generate_srcdict(ras, decs, fitsfiles_xx)
            srcdict_yy = self._generate_srcdict(ras, decs, fitsfiles_yy)
            srcdict = self._combine_srcdict(srcdict_xx, srcdict_yy)        
        self._srcdict_catdata(srcdict)
        self.pols = pols
        if return_data:
            return srcdict

    def _get_resolution(self, npix):
        grid_ha = np.linspace(0, np.pi, npix)
        return grid_ha[1] - grid_ha[0]

    def _get_npoints(self, npix):
        dr = self._get_resolution(npix)
        d_ha = self.ha_array[0, 0] - self.ha_array[0, -1]
        dn_ha = np.abs(d_ha) / dr
        return int(dn_ha) + 1

    def _interpolate_data(self, data, (azs, alts), (az0, alt0)):
        # linear interpolation
        dist = np.sqrt((azs - az0)**2 + (alts - alt0)**2)
        inds = np.argsort(dist)
        if dist[inds[0]] == 0:
            interp_data =  data[inds[0]]
        else:
            w1 = dist[inds[0]]**(-2)
            w2 = dist[inds[1]]**(-2)
            wgts = w1 + w2
            interp_data = (data[inds[0]] * w1 + data[inds[1]] * w2) / wgts
        return interp_data
  
    def interpolate_catalog(self, npix):
        data = self.data_array
        npoints = self._get_npoints(npix)
        data_array = np.zeros((len(self.pols), self.Nsrcs, npoints))
        azalt_array = np.zeros((2, self.Nsrcs, npoints))
        ha_array = np.zeros((self.Nsrcs, npoints))
        for ii in xrange(self.Nsrcs):
            azs = self.azalt_array[0, ii, :]
            alts = self.azalt_array[1, ii, :]
            ha_array[ii, :] = np.linspace(np.min(self.ha_array[ii, :]), np.max(self.ha_array[ii, :]), npoints)
            interp_azs, interp_alts = self._get_azalt(self.pos_array[ii][1], ha_array[ii, :])
            azalt_array[0, ii, :] = interp_azs
            azalt_array[1, ii, :] = interp_alts
            for jj, az in enumerate(interp_azs):
                for p in xrange(len(self.pols)):
                    data_array[p, ii, jj] = self._interpolate_data(data[p, ii, :], (azs, alts), (az, interp_alts[jj]))
        self.data_array = data_array
        self.azalt_array = azalt_array
        self.ha_array = ha_array
        self.Nfits = npoints

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
            mgp['pols'] = self.pols
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
            self.pols = mgp['pols'].value
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
