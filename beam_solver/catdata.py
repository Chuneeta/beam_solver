from collections import OrderedDict
import extract as et
import coord_utils as ct
import os, sys
import warnings
import h5py
import healpy
import numpy as np

pol2ind = {'xx':0, 'yy': 1}

class catData(object):
    """
    Object for stroing different catalog of celestial sources
    """
    def __init__(self):
        self.data_array = None
        self.az_alt_array = None
        self.ha_array = None
        self.pos_array = None
        self.err_array = None
        self.Nfits = None
        self.Nsrcs  = None
        self.Npols = None

    def get_unique(self, ras, decs, tol=2):
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
            same source. Default is 2 arcmin.
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
        for i, ra in enumerate(ras):
            pos_array[0, i] = ra; pos_array[1, i] = decs[i]
            key = (round(ra, 2), round(decs[i], 2))
            if not key in srcdict: srcdict[key] = {}
            for j in xrange(nfits):
                jdsplt = fitsfiles[j].split('/')[-1].split('.')
                jd = jdsplt[0] + '.' + jdsplt[1]
                srcstats = et.get_flux(fitsfiles[j], ra, decs[i], flux_type=flux_type)
		        # hack to add 5 mins since snapshots were phased to the middle timestamp
                lst, ha = ct.rajd2ha(ra, float(jd) + 5. / 60. / 24.)
                az, alt = ct.radec2azalt(float(jd) + 5. / 60. / 24., ra, decs[i])
                ha_array[i, j] = ha / 15.
                err_array[0, i, j] = srcstats['error']
                data_array[0, i, j] = srcstats['flux']
                azalt_array[0, i, j] = az
                azalt_array[1, i, j] = alt
            # saving to dictionary
            srcdict[key]['data'] = data_array
            srcdict[key]['error'] = err_array
            srcdict[key]['ha'] = ha_array
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

    def gen_polcatalog(self, fitsfiles_xx, fitsfiles_yy, ras, decs, flux_type='peak', return_data=False):
        """
        Extracts flux measurements at specified right ascension and declination values from the
        fitsfiles and generates a catData object containing the data and necessary metadata for
        xx and yy polarization. It can also return a dictionary containing the data and selected 
        metadata.

        Parameters
        ----------
        fitsfiles_xx : list of str
            List of of xx fitsfiles that will be used to generate or extract the source catalog.

        fitsfiles_yy : list of str
            List of of xx fitsfiles that will be used to generate or extract the source catalog.

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

        # checking if leghth of fitsxx is consistent with fitsyy
        assert len(fitsfiles_xx) == len(fitsfiles_yy), "Fitsfiles for xx and yy polarizations should be of the same length."
        # checking if length of ras is consistent with decs
        assert len(ras) == len(decs), "Right ascenscion array should be of the same size as declination array."
        nsrcs = len(ras)
        nfits = len(fitsfiles_xx)
        # initializating source dict and numpy arrays
        srcdict = OrderedDict()
        pos_array = np.ndarray([])
        ha_array = np.zeros((nsrcs, nfits))
        azalt_array = np.zeros((2, nsrcs, nfits))
        data_array = np.zeros((2, nsrcs, nfits))
        err_array = np.zeros((2, nsrcs, nfits))
        for i, ra in enumerate(ras):
            pos_array = np.append(pos_array, (ra, decs[i]))
            key = (round(ra, 2), round(decs[i], 2))
            if not key in srcdict: srcdict[key] = {}
            for j in xrange(nfits):
                jdspltxx = fitsfiles_xx[j].split('/')[-1].split('.')
                jdxx = jdspltxx[0] + '.' + jdspltxx[1]
                jdspltyy = fitsfiles_yy[j].split('/')[-1].split('.')
                jdyy = jdspltyy[0] + '.' + jdspltyy[1]
                if jdxx != jdxx:
                    print 'Skipping: mismatch in julian dates.'
                    continue
                srcstatsxx = et.get_flux(fitsfiles_xx[j], ra, decs[i])
                srcstatsyy = et.get_flux(fitsfiles_yy[j], ra, decs[i])
                # hack to add 5 mins since snapshots were phased to the middle timestamp
                lst, ha = ct.rajd2ha(ra, float(jdxx) + 5. / 60. / 24.)
                az, alt = ct.radec2azalt(float(jdxx) + 5. / 60. / 24., ra, decs[i])
                ha_array[i, j] = ha / 15.
                err_array[0, i, j] = srcstatsxx['error']
                err_array[1, i, j] = srcstatsyy['error']
                data_array[0, i, j] = srcstatsxx['flux']
                data_array[1, i, j] = srcstatsyy['flux']
                azalt_array[0, i, j] = az
                azalt_array[1, i, j] = alt
            # saving to dictionary
            srcdict[key]['data'] = data_array
            srcdict[key]['error'] = err_array
            srcdict[key]['ha'] = ha_array
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
            # read data
            dgp = f['Data']
            self.data_array = dgp['data_array'].value

    def calc_corrflux(self, beam, pol):
        """
        Calculates corrected flux values for all positions (ra, dec) using the measurements at different az-alt
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
        corr_array = np.ndarray((self.Nsrcs))
        beam_array = np.ndarray((self.Nfits))
        for i in range(self.Nsrcs):
            for j in range(self.Nfits):
                beam_array[j] = healpy.get_interp_val(beam, np.pi/2 - (alts[i, j] * np.pi/180.), azs[i, j] * np.pi/180.) 
            if self.data_array.shape[0] == 1:
                corr_array[i] = np.nansum(self.data_array[0, i, :] * beam_array) / np.nansum(beam_array ** 2)
            else:
                corr_array[i] = np.nansum(self.data_array[pol2ind[pol], i, :] * beam_array) / np.nansum(beam_array ** 2)
        return corr_array
