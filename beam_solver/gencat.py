import numpy as np
import extract as et
import coord_utils as ct
import pyuvdata

def read_cstbeam(cstfile):
    """
    Reads in beam model file and returns the beam map interpolated onto a healpix grid

    - cstfile : string
        CST file containing the beam model values
    """

    uvb = pyuvdata.UVBeam()
    uvb.read_cst_beam(cstfile, beam_type='power', frequency=150e6,
                  telescope_name='TEST', feed_name='bob',
                  feed_version='0.1', feed_pol='xx',
                  model_name='E-field pattern - Rigging height 4.9m',
                  model_version='1.0')
    uvb.peak_normalize()
    uvb.interpolation_function='az_za_simple'
    uvb.to_healpix()

    beamdata = uvb.data_array
    beamdata_xx = beamdata[0, 0, 0, 0, :]
    beamdata_yy = beamdata[0, 0, 1, 0, :]
    
    return beamdata_xx, beamdata_yy

def unique_sources(ra=[], dec=[]):
    """
    Selects only unique sources from given celestial coordinates
    
    Parameters
    ----------
    ra : lists of floats
        List containing right ascension values in degrees

    dec : list of floats
        List containing declination values in degrees

    """
    # checking if size of ra is consistent with size of dec
    assert ra.size == dec.size, "Length of ra should be consistent with length of dec"
    n = ra.size

    unq_ras = np.array([])
    unq_decs = np.array([])
    while n > 0:
        inds = np.array([])
        ra_c = np.array([])
        dec_c = np.array([])
        dist = np.sqrt(ras - ras[0]**2) + (decs - decs[0]**2)
        for ii, d in enumerate(dist):
            if d > tol:
                ra_c = np.append(ra_c, ra[ii])
                dec_c = np.append(dec_c, dec[ii])
                inds = np.append(inds, ii) 
        unq_ra = np.append(unq_ra, np.mean(ra_c)
        unq_dec = np.append(unq_dec, np.mean(dec_c)
        ra = np.delete(ra, inds)
        dec = np.delete(dec, inds)
        n = len(ras)

    return unq_ras, unq_decs

def gen_catologue(fitsfiles=[], ras=[], decs=[], outfile=None):
    """
    Extracts flux measurements from sources
    
    Parameters
    ----------

    """
    for ii, ra in enumerate(ras):
        for fits in fitsfiles:
            srcdict = get_flux(fits, ra, decs[ii])
            cra, cdec = ft.get_centre_radec(fitsfile)
            lst = cra
            ha = lst - ra
            az, alt = ct.hadec2azalt(ha, decs[ii], lat)
            
            
    
                                 	        	                              
         

    
