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

class genCatalog(object):
    def __init__(self, fitsxx=[], fitsyy=[], ras=[], decs=[], beam=None):
        """
        Object to stored the images/fitsfiles and generating a catalogue of astronomical/celestial sources

        Parameters
        ----------
        fitsxx : list of str
            Name of fitsfiles or snapshots that will be used to generate the catalogue of sources. Default is empty list.
    
	fitsyy : list of str

        ras : list of floats
            Right ascension values in degrees to use to extract the flux densities for generating the catalog. Default is empty list.

        dec : list of floats
            Declination values in degrees to use to extract the flux densities for generating the catalog. Default is empty list.

        beam : catBeam object
            catBeam object containing information about the primary beam. Default is None
        """

        self.fitsxx = fitsxx
        self.fitsyy = fitsyy
        self.ras = ras
        self.decs = decs
        self.unique_ras = ras
        self.unique_decs = decs
        self.beam = beam        

    def get_unique_sources(self, tol=2):
        """
        Selects only unique sources from given celestial coordinates

        Parameters
        ----------
        tol : float
            Tolerance or radius in arcmin within which a source might be considered as the 
            same source. Default is 2 arcmin.
        """

        # checking if size of ra is consistent with size of dec
        ras = self.ras; decs = self.decs
        assert len(ras) == len(decs), "Length of ras should be consistent with length of decs"
        n = len(ras)

        if isinstance(ras, list): ras = np.array(ras)
        if isinstance(decs, list): decs = np.array(decs)

        unq_ras = np.array([])
        unq_decs = np.array([])
        while n > 0:
            inds = np.array([])
            ra_c = np.array([])
            dec_c = np.array([])
            dist = np.sqrt((ras - ras[0])**2 + (decs - decs[0])**2) * 60
            for ii, d in enumerate(dist):
                if d < tol:
                    ra_c = np.append(ra_c, ras[ii])
                    dec_c = np.append(dec_c, decs[ii])
                    inds = np.append(inds, ii) 
            unq_ras = np.append(unq_ras, np.mean(ra_c))
            unq_decs = np.append(unq_decs, np.mean(dec_c))
            ras = np.delete(ras, inds)
            decs = np.delete(decs, inds)
            n = len(ras)
        
        self.unique_ras = unq_ras; self.unique_decs= unq_decs
        print ('Found {} unique sources out of {}.'.format(len(unq_ras), len(self.ras)))

    def gen_catologue(self):
        """
        Extracts flux measurements from sources
        """
        nsrc = len(self.unique_ras)
        nfitsxx = len(self.fitsxx)
        nfitsyy = len(self.fitsyy)

        if nfitsxx != nfitsyy:
            warnings.warn('WARNING: Number of images for both polarization are not the same, thus the values for the missing ones will be set to 0', Warning)
            if nfitsxx > nfitsyy:
                nfits = nfitsxx
            else:
                nfits = nfitsyy
        else:
            nfits = nfitsxx

        has = np.zeros((nsrc, nfits))
        lsts = np.zeros((nsrc, nfits))
        jds = np.zeros((nsrc, nfits))
        azalts = np.zeros((2, nsrc, nfits)) 
        pfluxs = np.zeros((2, nsrc, nfits))
        tfluxs = np.zeros((2, nsrc, nfits))
        rms = np.zeros((2, nsrc, nfits))
        beamvals = np.zeros((2, nsrc, nfits))
        freqs = np.zeros((2, nsrc, nfits))
        pcorrfluxs = np.zeros((2, nsrc))
        tcorrfluxs = np.zeros((2, nsrc))

	print nfits, nfitsxx, nfitsyy
        # beam model values
        for i, ra in enumerate(self.unique_ras):
            for j in xrange(nfits):
                # hack for the time being to get the julian date, the fits files are named after the julian dates
                jdspltxx = self.fitsxx[j].split('/')[-1].split('.')
                jdxx = jdspltxx[0] + '.' + jdspltxx[1]
                jdspltyy = self.fitsyy[j].split('/')[-1].split('.')
                jdyy = jdspltyy[0] + '.' + jdspltyy[1]
                if jdxx != jdyy : continue
                srcstatsxx = et.get_flux(self.fitsxx[i], ra, self.unique_decs[i])
                srcstatsyy = et.get_flux(self.fitsyy[i], ra, self.unique_decs[i])
                # hack to add 5 mins since snapshots were phased to the middle timestamp
                lst, ha = ct.rajd2ha(ra, float(jdxx) + 5./60./24.)
                az, alt = ct.radec2azalt(float(jdxx) + 5./60./24., ra, self.unique_decs[i])
                beamvalxx = healpy.get_interp_val(self.beam['data']['xx'], np.pi/2 - alt * np.pi/180., az * np.pi/180.)
                beamvalyy = healpy.get_interp_val(self.beam['data']['yy'], np.pi/2 - alt * np.pi/180., az * np.pi/180.)
                has[i, j] = ha / 15.
                lsts[i, j] = lst / 15.
                freqs[0, i, j] = srcstatsxx['freq']
		freqs[1, i, j] = srcstatsyy['freq']
                rms[0, i, j] = srcstatsxx['rms']
		rms[1, i, j] = srcstatsyy['rms']
                pfluxs[0, i, j] = srcstatsxx['peak']
                pfluxs[1, i, j] = srcstatsyy['peak']
                tfluxs[0, i, j] = srcstatsxx['gauss_int']
                tfluxs[1, i, j] = srcstatsyy['gauss_int']
                azalts[0, i, j] = az 
                azalts[1, i, j] = alt
                beamvals[0, i, j] = beamvalxx
                beamvals[1, i, j] = beamvalyy 
            # calculating corrected flux values
            pcorrfluxs[0, i] = np.nansum(pfluxs[0, i, :] * beamvals[0, i, :]) / np.nansum(beamvals[0, i, :]**2)
            pcorrfluxs[1, i] = np.nansum(pfluxs[1, i, :] * beamvals[1, i, :]) / np.nansum(beamvals[1, i, :]**2)
            tcorrfluxs[0, i] = np.nansum(tfluxs[0, i, :] * beamvals[0, i, :]) / np.nansum(beamvals[0, i, :]**2)
            tcorrfluxs[1, i] = np.nansum(tfluxs[1, i, :] * beamvals[1, i, :]) / np.nansum(beamvals[1, i, :]**2)

        # fill up srcdata object
        srcd = catdata.catData()
        srcd.pflux_array = pfluxs
        srcd.tflux_array = tfluxs
        srcd.pcorr_array = pcorrfluxs
        srcd.tcorr_array = tcorrfluxs

        # fill in metadata
        srcd.ras = self.unique_ras
        srcd.decs = self.unique_decs
        srcd.ha_array = has
        srcd.lst_array = lsts
        srcd.rms_array = rms
        srcd.freq_array = np.unique(freqs)
        srcd.jd_array = np.unique(jds)
        srcd.azalt_array = azalts
        srcd.beam_array = beamvals
        srcd.Nfits = nfits
        srcd.Nsrcs = nsrc
        srcd.beam_type = self.beam['beam_type']
        srcd.beam_size = self.beam['size']
        srcd.beam_normalization = self.beam['normalization']

        return srcd
    """
        for ii, ra in enumerate(self.unique_ras):
            for jj, fits in enumerate(self.fitsfiles):
                jdsplt = fits.split('/')[-1].split('.')
                jd = jdsplt[0] + '.' + jdsplt[1]
                srcstats = et.get_flux(fits, ra, self.unique_decs[ii])
                #cra, cdec = et.get_centre_radec(fits)
                #lst = cra
                #ha = lst - ra
                lst, ha = ct.rajd2ha(ra, float(jd) + 5./60./24.)
                #az, alt = ct.hadec2azalt(ha, self.unique_decs[ii], hera_lat)
                az, alt = ct.radec2azalt(float(jd) + 5./60./24., ra, self.unique_decs[ii])
                beamval = healpy.get_interp_val(self.beam['data'], np.pi/2 - alt * np.pi/180., az * np.pi/180.)
                has[ii, jj] = ha / 15.
                lsts[ii, jj] = lst / 15.
                pfluxs[ii, jj] = srcstats['peak']
                tfluxs[ii, jj] = srcstats['gauss_int']
                freqs[ii, jj] = srcstats['freq']
                rms[ii, jj] = srcstats['rms']
                azalts[0, ii, jj] = az ; azalts[1, ii, jj] = alt 
                beamvals[ii, jj] = beamval
"""

