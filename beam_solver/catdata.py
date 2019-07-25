from collections import OrderedDict, Counter
from beam_solver import extract as et
from beam_solver import coord_utils as ct
from beam_solver import fits_utils as ft
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
        self.error_array = None
        self.azalt_array = None
        self.ha_array = None
        self.pos_array = None
        self.Nfits = None
        self.Nsrcs  = None
        self.pols = None

    def _get_attr(self, attr):
        return getattr(self, attr)

    def _set_attr(self, attr, value):
        setattr(self, attr, value)        

    def _getattrs(self):
        ha_array = self._get_attr('ha_array')
        data_array = self._get_attr('data_array')
        error_array = self._get_attr('error_array')
        pos_array = self._get_attr('pos_array')
        azalt_array = self._get_attr('azalt_array')
        nsrcs = self._get_attr('Nsrcs')
        return data_array, error_array, azalt_array, ha_array, pos_array, nsrcs

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
        if isinstance(ha, list):
            ha[ha == 0.] = 1e-7
        az, alt = ct.hadec2azalt(dec * np.pi/180., ha)
        return az, alt

    def _generate_srcdict(self, ras, decs, fitsfiles, flux_type):
        """
        Extracts flux measurements at specified right ascension and declination values from the fitsfiles
        and return dictionary containing the data and necessary metadata for single polarization.
        polarization. 
        Parameters
        ---------
        ras : list of float
            List of right ascension values in degrees.
        decs : list of floats
            List of declination values in degrees.
        fitsfiles : list of str
            List of of xx or yy fitsfiles that will be used to extract the flux values.
        flux_type : str
            Flux type to extract, options: 
                'pflux'       : peak value from the data
                'gauss_pflux' : peak value of the 2d gaussian fit of the data
                'gauss_tflux' : total integrated value of the 2d gaussian fit of the data                        """
        # selecting unique ras and decs
        nsrcs = len(ras)
        nfits = len(fitsfiles)
        srcdict = OrderedDict()
        ha_array = np.zeros((nsrcs, nfits))
        error_array = np.zeros((nsrcs, nfits))
        data_array = np.zeros((nsrcs, nfits))
        azalt_array = np.zeros((2, nsrcs, nfits))
        jds = self._get_jds(fitsfiles)
        for i, ra in enumerate(ras):
            key = (round(ra, 2), round(decs[i], 2))
            if not key in srcdict: srcdict[key] = {}
            for j, fn in enumerate(fitsfiles):
                srcstats = et.get_flux(fn, ra, decs[i])
                lst, ha = self._get_lstha(jds[j], ra)
                az, alt = self._get_azalt(decs[i], ha)
                ha_array[i, j] = ha
                error_array[i, j] = srcstats['error']
                data_array[i, j] = srcstats[flux_type]
                azalt_array[0, i, j] = az
                azalt_array[1, i, j] = alt
                # saving to dictionary
            srcdict[key]['data'] = data_array[i,  :]
            srcdict[key]['error'] = error_array[i, :]
            srcdict[key]['ha'] = ha_array[i, :]
            srcdict[key]['azalt'] = azalt_array[:, i, :]
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
        keys = list(srcdict.keys())
        self.pos_array = keys
        _sh1 = len(keys)
        _sh = srcdict[keys[0]]['data'].shape
        if len(_sh) == 1:
            _sh0 = 1; _sh2 = _sh[0]
        else:
            _sh0 = _sh[0]; _sh2 = _sh[1]  
        self.data_array = np.zeros((_sh0, _sh1, _sh2)) 
        self.error_array = np.zeros((_sh0, _sh1, _sh2))
        self.ha_array = np.zeros((_sh1, _sh2))
        self.azalt_array= np.zeros((2, _sh1, _sh2))
        for i, key in enumerate(keys):
            self.data_array[:, i, :] = srcdict[key]['data']
            self.error_array[:, i, :] = srcdict[key]['error']
            self.ha_array[i, :] = srcdict[key]['ha']
            self.azalt_array[:, i, :] = srcdict[key]['azalt']
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
        keys_xx = list(srcdict_xx.keys())
        keys_yy = list(srcdict_yy.keys())
        srcdict = copy.deepcopy(srcdict_xx)
        assert keys_xx == keys_yy, "both dictionary should have the same keywords."
        for key in keys_xx:
            for skey in ['data']:
                srcdict[key][skey] = np.array([srcdict_xx[key][skey], srcdict_yy[key][skey]])
        return srcdict

    def gen_catalog(self, ras, decs, fitsfiles_xx=None, fitsfiles_yy=None, pols='xx', flux_type='pflux', return_data=False):
        """
        Extracts flux measurements at specified right ascension and declination values from the fitsfiles
        and generates a catdata object containing the data and necessary metadata for xx or yy or both
        polarization. It can also return a dictionary containing the data and selected metadata.
        Parameters
        ---------
        ras: list of float
            List of right ascension values in degrees.
        decs: list of floats
            List of declination values in degrees.
        fitsfiles_xx: list of str
            List of of xx fitsfiles that will be used to generate or extract the source catalog.
        fitsfiles_yy: list of str
            List of of yy fitsfiles that will be used to generate or extract the source catalog.
        pols: str ot list of str
            Polizations can be xx or yy or both.
        flux_type: str
            Flux type to extract, options:
                'pflux'       : peak value from the data
                'gauss_pflux' : peak value of the 2d gaussian fit of the data
                'gauss_tflux' : total integrated value of the 2d gaussian fit of the data
        return_data: boolean
            If True, returns dictionary with the data values and selected metadata.
        """
        assert len(ras) == len(decs), "Right ascenscion array should be of the same size as declination array."
        if not isinstance(pols, list): pols = [pols]
        npols = len(pols)
        if npols == 1:
            fitsfiles = fitsfiles_xx if pols[0] == 'xx' else fitsfiles_yy 
            srcdict = self._generate_srcdict(ras, decs, fitsfiles, flux_type) 
        else:
            srcdict_xx = self._generate_srcdict(ras, decs, fitsfiles_xx, flux_type)
            srcdict_yy = self._generate_srcdict(ras, decs, fitsfiles_yy, flux_type)
            srcdict = self._combine_srcdict(srcdict_xx, srcdict_yy)        
        self._srcdict_catdata(srcdict)
        self.pols = pols
        if return_data:
            return srcdict

    def add_src(self, ras, decs, fitsfiles_xx=None, fitsfiles_yy=None, pols='xx', flux_type='pflux'):
        """
        Adding data and metadata to an existing catalog.
        Parameters
        ----------
        ras: list of float
            List of right ascension values in degrees.
        decs: list of floats
            List of declination values in degrees.
        fitsfiles_xx: list of str
            List of of xx fitsfiles that will be used to generate or extract the source catalog.
        fitsfiles_yy: list of str
            List of of yy fitsfiles that will be used to generate or extract the source catalog.
        pol: str ot list of str
            Polizations can be xx or yy or both.
        flux_type: str
            Flux type to extract, options:
                'pflux'       : peak value from the data
                'gauss_pflux' : peak value of the 2d gaussian fit of the data
                'gauss_tflux' : total integrated value of the 2d gaussian fit of the data
        """ 
        if not fitsfiles_xx is None:
            assert len(fitsfiles_xx) == self.Nfits, "Number of fitsfiles is not consistent with Nfits."
        if not fitsfiles_yy is None:
            assert len(fitsfiles_yy) == self.Nfits, "Number of fitsfiles is not consistent with Nfits."
        if not isinstance(pols, list): pols = list(pols)
        assert len(pols) == len(self.pols), "Pols need to be consistent with self.pols"
        data_array, error_array, azalt_array, ha_array, pos_array, nsrcs = self._getattrs()
        self.gen_catalog(ras, decs, fitsfiles_xx=fitsfiles_xx, fitsfiles_yy=fitsfiles_yy, pols=pols, flux_type=flux_type)
        self._set_attr('data_array', np.append(data_array, self.data_array, axis=1))
        self._set_attr('error_array', np.append(error_array, self.error_array, axis=1))
        self._set_attr('azalt_array', np.append(azalt_array, self.azalt_array, axis=1))
        self._set_attr('ha_array', np.append(ha_array, self.ha_array, axis=0))
        self._set_attr('pos_array', np.append(pos_array, self.pos_array, axis=0))
        self._set_attr('Nsrcs', nsrcs + len(ras))

    def _get_npoints(self, dr):
        """
        Calculates number of data points for desired spacing
        Parameters:
        ---------- 
        dr: float
            Spacing between two consecutive points
        """ 
        d_ha = self.ha_array[0, 0] - self.ha_array[0, -1]
        dn_ha = np.abs(d_ha) / dr
        return int(dn_ha) + 1

    def _interpolate_data(self, x, y, kind, bounds_error):
        """
        1-D interpolation function
        Parameters
        ----------
        x: np.ndarray
            A 1-D array of real values.
        y: np.ndarray
            A N-D array of real values.
            The length of `y` along the interpolation axis must be equal to the length of `x`.
        kind: str, optional
            Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 
            'slinear', 'quadratic', 'cubic','previous', 'next', where 'zero', 'slinear', 
            'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second 
            or third order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline interpolator 
            to use. Default is 'linear'.
        bounds_error: boolean, optional
            If True, a ValueError is raised any time interpolation is attempted on a value outside
            of the range of x (where extrapolation is necessary). If False, out of bounds values
            are assigned `fill_value`.
            By default, an error is raised unless `fill_value="extrapolate"`.
        """
        return interpolate.interp1d(x.compress(~np.isnan(y)), y.compress(~np.isnan(y)), kind=kind, bounds_error=bounds_error)
 
    def interpolate_catalog(self, dha=0.01, kind='cubic', discard_neg=False, bounds_error=True):
        """
        Interpolates the points between source tracks using any interpolation algorithm. Default one uses
        cubic interpolation.
        Parameters
        ----------
        dha: float
            Spacing in hour angle on which the interpolation is carried out. The spacing should be
            specified in radians. Default is 0.01.
        kind: str
             Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero',
            'slinear', 'quadratic', 'cubic','previous', 'next', where 'zero', 'slinear',
            'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second
            or third order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline interpolator
            to use. Default is 'linear'.
            Interpolation algorithm, can be 'linear, cubic, nearest'. Default is cubic interpolation.
        discard_neg: boolean
            True discards all negative data values during interpolation. Default is False.
        bounds_errors: boolean
            If True, a ValueError is raised any time interpolation is attempted on a value outside
            of the range of x (where extrapolation is necessary). If False, out of bounds values 
            are assigned `fill_value`.
            By default, an error is raised unless `fill_value="extrapolate"`.
        """
        data = self.data_array
        npoints = self._get_npoints(dha)
        data_array = np.zeros((len(self.pols), self.Nsrcs, npoints))
        error_array = np.zeros((len(self.pols), self.Nsrcs, npoints))
        azalt_array = np.zeros((2, self.Nsrcs, npoints))
        ha_array = np.zeros((self.Nsrcs, npoints))
        for i in range(self.Nsrcs):
            for p in range(len(self.pols)):
                # discarding data points with zero to avoid jump in the interpolation
                if discard_neg:
                    ind0 = np.where(self.data_array[p, i, :] > 0)
                else:
                    ind0 = np.arange(len(self.data_array[p, i, :]))
                data = self.data_array[p, i, ind0]
                error = self.error_array[p, i, ind0]
                ha = self.ha_array[i, ind0]
                ha_array[i, :] = np.linspace(np.min(ha), np.max(ha), npoints)
                interp_azs, interp_alts = self._get_azalt(self.pos_array[i][1], ha_array[i, :])
                azalt_array[0, i, :] = interp_azs
                azalt_array[1, i, :] = interp_alts
                # interpolating data and error
                interp_func_d = self._interpolate_data(ha, data, kind=kind, bounds_error=bounds_error)
                interp_func_e = self._interpolate_data(ha, error, kind=kind, bounds_error=bounds_error)
                data_array[p, i, :] = interp_func_d(ha_array[i, :])
                error_array[p, i, :] = interp_func_e(ha_array[i, :])
    
        self.data_array = data_array
        self.azalt_array = azalt_array
        self.error_array = error_array
        self.ha_array = ha_array
        self.Nfits = npoints

    def delete_src(self, keys):
        """
        Deletes sources given key. The key is in the form (ra, dec) in degrees.
        The right ascension and declination values can be found in pos_array
        keys : tuple or list of tuples
            Key should a tuple in the form of (ra, dec).
        """
        if not isinstance(keys, list): keys = [keys]
        for key in keys:
            ind = np.where(self.pos_array == [key])
            if len(ind[0]) == 0:
                raise ValueError('{} could be not found.'.format(key))
            if len(ind[0]) > 2:
                counter = Counter(ind[0])
                ind0 = [counter.most_common(1)[0][0]] * 2
            else:
                ind0 = ind[0]    
            self.data_array = np.delete(self.data_array, ind0, 1)
            self.error_array = np.delete(self.error_array, ind0, 1)
            self.ha_array = np.delete(self.ha_array, ind0, 0)
            self.pos_array = np.delete(self.pos_array, ind0, 0)
            self.azalt_array = np.delete(self.azalt_array, ind0, 1)
            self.Nsrcs -= 1

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
            mgp['pols'] = [np.string_(p) for p in self.pols] 
            mgp['ha_array'] = self.ha_array
            mgp['error_array'] = self.error_array
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
            self.error_array = mgp['error_array'].value
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
