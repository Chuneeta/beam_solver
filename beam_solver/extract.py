"""
Functions to extract fluxes from the fits files
"""
from astropy import wcs
from astropy.io import fits

def get_fitsinfo(fitsname):
    """
    Read fitsfile and extracts necessary information

    Parameters
    ----------
    fitsname : string
        Name of input fitsfile
    """

    data, header = fits.getdata(fitsname, header=True)
    data = data.squeeze()
    freq = header['CRVAL3']
    dra = header['CDELT1']
    ddec = header['CDELT2']
    naxis = header['NAXIS1']

    return {'data':data, 'header':hdr, 'freq':freq, 'naxis':naxis}
    

def 
