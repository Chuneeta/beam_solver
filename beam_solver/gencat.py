import numpy as np
import extract as et
import coord_utils as ct
import pyuvdata
import warnings
import healpy
import catdata

# latitutude and liongitude of HERA in degrees
hera_lat = -30.72138888888889
hera_lon = 21.011133333333333


def get_unique(ras=[], decs=[], tol=2):
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

    # checking if ras and decs are empty
    if len(ras) == 0:
        raise ValueError('Array for right ascension values is empty.')
    if len(decs) == 0:
        raise ValueError('Array for declination values is empty.')

    # checking if size of ra is consistent with size of dec
    assert len(ras) == len(decs), "Length of ras should be consistent with length of decs."
    n0 = len(ras)
    n = n0

    if isinstance(ras, list):
        ras = np.array(ras, dtype=np.float32)
    if isinstance(decs, list):
        decs = np.array(decs)

    unq_ras = np.array([])
    unq_decs = np.array([])
    while n > 0:
        inds = np.array([])
        ra_c = np.array([])
        dec_c = np.array([])
        # calculating the distance between adjacent coordinates (ra and dec) in arcmins
        dist = np.sqrt((ras - ras[0])**2 + (decs - decs[0])**2) * 60
        
        for ii, d in enumerate(dist):
            if d < tol:  # checking if it is with the specified tolearance value
                ra_c = np.append(ra_c, ras[ii])
                dec_c = np.append(dec_c, decs[ii])
                inds = np.append(inds, ii)
                unq_ras = np.append(unq_ras, np.mean(ra_c))
                unq_decs = np.append(unq_decs, np.mean(dec_c))
                ras = np.delete(ras, inds)
                decs = np.delete(decs, inds)
                n = len(ras)

    print ('Found {} unique sources out of {}.'.format(len(unq_ras), n0))
    return unq_ras, unq_decs


class genCatBase(object):
    def __init__(self, fits=[], ras=[], decs=[]):
        """
        Object to store either xx or yy polarization images/fitsfiles and generating a catolgue
        of sources.

        Parameters
        ----------
        fits : list of str
            Name of xx or yy fitsfiles or snapshots that wull be used to generate the catalogue
            the sources.

        ras : list of floats
            List of right ascension values in degrees to use to extract the flux densities for
            generating the catalog. Default list is empty.

        decs : list of floats
            List of declination values in degrees to use to extract the flux densities for
            generating the catalog. Default list is empty.
        """

        # selecting only uniques right ascension and declination values
        self.ras = ras
        self.decs = decs
        self.fits = fits

    def check(self):
        """
        Checks if the inputs are proper
        """

        # convert to list if inputs are given as a string, float or integer
        if not isinstance(self.fits, list):
            if isinstance(self.fits, np.ndarray):
                self.fits.tolist()
            else:
                self.fits = [self.fits]

        if not isinstance(self.ras, list):
            if isinstance(self.ras, np.ndarray):
                self.ras.tolist()
            else:
                self.ras = [self.ras]

        if not isinstance(self.decs, list):
            if isinstance(self.decs, np.ndarray):
                self.decs.tolist()
            else:
                self.decs = [self.decs]

        # checking type of the inputs
        if all(isinstance(elm, (str, np.str)) for elm in self.fits) is False:
            raise (ValueError, 'Fitsfiles should be strings.')

        if all(isinstance(elm, (int, np.int, float, np.float)) for elm in self.ras) is False:
            raise (ValueError, 'Right ascension values should be float of integers.')

        if all(isinstance(elm, (int, np.int, float, np.float)) for elm in self.decs) is False:
            raise (ValueError, 'Right ascension values should be float of integers.')

    def gen_catalog(self):
        """
        Extracts flux measurements at specified right ascension and declination values from the
        fitsfiles and generates a catdata object containing the data and necessary metadata for
        xx or yy polarization.
        """

        self.check()
        self.ras, self.decs = get_unique(self.ras, self.decs)

        nsrc = len(self.ras)
        nfits = len(self.fits)

        ha_array = np.zeros((nsrc, nfits))
        lst_array = np.zeros((nsrc, nfits))
        jd_array = np.zeros((nsrc, nfits))
        azalt_array = np.zeros((2, nsrc, nfits))
        pflux_array = np.zeros((1, nsrc, nfits))
        tflux_array = np.zeros((1, nsrc, nfits))
        rms_array = np.zeros((1, nsrc, nfits))
        freq_array = np.zeros((nsrc, nfits))

        for i, ra in enumerate(self.ras):
            # checking ra and dec inputs
            if (ra > 360) | (ra < 0):
                raise ValueError('Right ascension value should be 0 =< ra < 360.')

            if (self.decs[i] > 0):
                raise ValueError('Declination should be negative as the longitude of HERA is around -30 degrees (Southern Hemishere.')

            for j in xrange(nfits):
                jdsplt = self.fits[j].split('/')[-1].split('.')
                jd = jdsplt[0] + '.' + jdsplt[1]
                srcstats = et.get_flux(self.fits[j], ra, self.decs[i])
                # hack to add 5 mins since snapshots were phased to the middle timestamp
                lst, ha = ct.rajd2ha(ra, float(jd) + 5. / 60. / 24.)
                az, alt = ct.radec2azalt(float(jd) + 5. / 60. / 24., ra, self.decs[i])
                ha_array[i, j] = ha / 15.
                lst_array[i, j] = lst / 15.
                jd_array[i, j] = jd
                freq_array[i, j] = srcstats['freq']
                rms_array[0, i, j] = srcstats['rms']
                pflux_array[0, i, j] = srcstats['peak']
                tflux_array[0, i, j] = srcstats['gauss_int']
                azalt_array[0, i, j] = az
                azalt_array[1, i, j] = alt

        # fill up srcdata object
        srcd = catdata.catData()
        srcd.pflux_array = pflux_array
        srcd.tflux_array = tflux_array

        # fill in metadata
        srcd.ras = self.ras
        srcd.decs = self.decs
        srcd.ha_array = ha_array
        srcd.lst_array = lst_array
        srcd.rms_array = rms_array
        srcd.freq_array = np.unique(freq_array)
        srcd.jd_array = np.unique(jd_array)
        srcd.azalt_array = azalt_array
        srcd.Nfits = nfits
        srcd.Nsrcs = nsrc
        srcd.Npols = 1
	
        # optional attributes
        srcd.beam_array = np.zeros((1, nsrc, nfits))
        srcd.pcorr_array = np.zeros((1, nsrc,))
        srcd.tcorr_array = np.zeros((1, nsrc,))
        srcd.beam_size = 0
        srcd.beam_type = 'None'
        srcd.beam_normalization = 'None'

        return srcd

class genCatCross(object):
    def __init__(self, fitsxx=[], fitsyy=[], ras=[], decs=[]):
        """
        Object to store both xx and yy polarization images/fitsfiles and generating a catolgue
        of sources

        Parameters
        ----------
        fitsxx : list of str
            List of xx fitsfiles or snapshots that will be used to generate the catalogue
            of sources.

        fitsyy : list of str
            List of yy fitsfiles or snpashots that will be used to generate the catalog of
            source.

        ras : list of floats
            List of right ascension values in degrees to use to extract the flux densities for
            generating the catalog. Default list is empty.

        decs : list of floats
            List of declination values in degrees to use to extract the flux densities for
            generating the catalog. Default list is empty.
        """

        self.ras = ras
        self.decs = decs
        self.fitsxx = fitsxx
        self.fitsyy = fitsyy

    def check(self):
        """
        Checks if the inputs are proper.
        """

        # convert to list if inputs are given as a string, float or integer
        if not isinstance(self.fitsxx, list):
            self.fitsxx = [self.fitsxx]

        if not isinstance(self.fitsyy, list):
            self.fitsyy = [self.fitsyy]

        if not isinstance(self.ras, list):
            self.ras = [self.ras]

        if not isinstance(self.decs, list):
            self.decs = [self.decs]

        # checking type of the inputs
        if all(isinstance(elm, (str, np.str)) for elm in self.fitsxx) is False:
            raise (ValueError, 'Fitsfiles should be strings.')

        if all(isinstance(elm, (str, np.str)) for elm in self.fitsyy) is False:
            raise (ValueError, 'Fitsfiles should be strings.')

        if all(isinstance(elm, (int, np.int, float, np.float)) for elm in self.ras) is False:
            raise (ValueError, 'Right ascension values should be float ot integers')

        if all(isinstance(elm, (int, np.int, float, np.float)) for elm in self.decs) is False:
            raise (ValueError, 'Right ascension values should be float ot integers')

        # checking if number of fitsfiles are consistent
        assert len(self.fitsxx) == len(self.fitsyy), 'Number of fitsfiles for both polarizations should be the same.'

    def gen_catalog(self):
        """
        Extracts flux measurements at specified right ascension and declination values from the
        xx and yy fitsfiles and generates a catdata object containing the data and necessary metadata for
        both xx and yy polarization.
        """

        self.check()
        self.ras, self.decs = get_unique(self.ras, self.decs)

        nsrc = len(self.ras)
        nfitsxx = len(self.fitsxx)
        nfitsyy = len(self.fitsyy)

        ha_array = np.zeros((nsrc, nfitsxx))
        lst_array = np.zeros((nsrc, nfitsxx))
        jd_array = np.zeros((nsrc, nfitsxx))
        azalt_array = np.zeros((2, nsrc, nfitsxx))
        pflux_array = np.zeros((2, nsrc, nfitsxx))
        tflux_array = np.zeros((2, nsrc, nfitsxx))
        rms_array = np.zeros((2, nsrc, nfitsxx))
        freq_array = np.zeros((nsrc, nfitsxx))

        for i, ra in enumerate(self.ras):
            # checking ra and dec inputs
            if (ra > 360) | (ra < 0):
                raise ValueError('Right ascension value should be 0 =< ra < 360.')

            if (self.decs[i] > 0):
                raise ValueError('Declination should be negative as the longitude of HERA is around -30 degrees (Southern Hemishere.')

            for j in xrange(nfitsxx):
                jdspltxx = self.fitsxx[j].split('/')[-1].split('.')
                jdxx = jdspltxx[0] + '.' + jdspltxx[1]
                jdspltyy = self.fitsyy[j].split('/')[-1].split('.')
                jdyy = jdspltyy[0] + '.' + jdspltyy[1]
                if jdxx != jdyy:
                    print ('SKIPPING: Julian dates for the polarizations are not consistent.')
                    continue
                # checking if jds are consistent for both polarizations
                srcstatsxx = et.get_flux(self.fitsxx[j], ra, self.decs[i])
                srcstatsyy = et.get_flux(self.fitsyy[j], ra, self.decs[i])
                # hack to add 5 mins since snapshots were phased to the middle timestamp
                lst, ha = ct.rajd2ha(ra, float(jdxx) + 5. / 60. / 24.)
                az, alt = ct.radec2azalt(float(jdxx) + 5. / 60. / 24., ra, self.decs[i])
                ha_array[i, j] = ha / 15.
                lst_array[i, j] = lst / 15.
                # checking if frequencies are consistent for both polrizations
                if srcstatsxx['freq'] != srcstatsyy['freq']:
                    print ('SKIPPING: Frequency for the polarizations are not consistent.')
                    continue
                freq_array[i, j] = srcstatsxx['freq']
                rms_array[0, i, j] = srcstatsxx['rms']
                rms_array[1, i, j] = srcstatsyy['rms']
                pflux_array[0, i, j] = srcstatsxx['peak']
                pflux_array[1, i, j] = srcstatsyy['peak']
                tflux_array[0, i, j] = srcstatsxx['gauss_int']
                tflux_array[1, i, j] = srcstatsyy['gauss_int']
                azalt_array[0, i, j] = az
                azalt_array[1, i, j] = alt

        # fill up srcdata object
        srcd = catdata.catData()
        srcd.pflux_array = pflux_array
        srcd.tflux_array = tflux_array

        # fill in metadata
        srcd.ras = self.ras
        srcd.decs = self.decs
        srcd.ha_array = ha_array
        srcd.lst_array = lst_array
        srcd.rms_array = rms_array
        print freq_array.shape
        srcd.freq_array = freq_array[0, :]
        srcd.jd_array = np.unique(jd_array)
        srcd.azalt_array = azalt_array
        srcd.Nfits = nfitsxx
        srcd.Nsrcs = nsrc

        return srcd
