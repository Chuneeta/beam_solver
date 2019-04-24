import numpy as np
import ephem
from astropy.time import Time

# setting HERA observer
hera_lat = '-30:43:17'
hera_lon = '21:25:40.08'
hera = ephem.Observer()
hera.lat, hera.long, hera.elevation = hera_lat, hera_lon, 0.0
j0 = ephem.julian_date(0)

def deg2hms(ra):
    """
    Converts degrees to hours minutes seconds
    ra : right ascension or any value to convert to hours minutes seconds
    """
    ra_hrs = ra / 15.
    assert ra_hrs >= 0.0, "right ascension or value should be positive"
    ra_h = int(ra_hrs)
    ra_mins = (ra_hrs - ra_h) * 60
    ra_m = int(ra_mins)
    ra_secs = (ra_mins - ra_m) * 60
    ra_s = round(ra_secs, 2) 
    return '{}h{}m{}s'.format(ra_h, ra_m, ra_s)

def deg2dms(dec):
    """
    Converts to degrees to degrees minutes seconds
    dec : declination or any other value to convert to degrees minutes seconds
    """
    dec_a = np.abs(dec)
    dec_deg = int(dec_a)
    dec_mins =  (dec_a - dec_deg) * 60
    dec_m = int(dec_mins)
    dec_secs = (dec_mins - dec_m) * 60
    dec_s = round(dec_secs, 2)
    sign = np.sign(dec)
    return '{}d{}m{}s'.format(int(sign * dec_deg), dec_m, dec_s)

def hms2deg(hms_str):
    """
    Converts hours minutes seconds to degrees
    hms_str : string specifying hours minutes seconds in the format hrs:min:sec
    """
    str_splt = map(float, hms_str.split(':'))
    assert str_splt[0] >= 0, "hours needs to be positive quantity"
    assert str_splt[1] >= 0, "minutes needs to be positive quantity"
    assert str_splt[2] >= 0, "seconds needs to be positive quantity"
    hrs = str_splt[0] + str_splt[1]/60. + str_splt[2]/3600. 
    deg = hrs * 15
    return round(deg, 2)

def dms2deg(dms_str):
    """
    Converts degrees minutes seconds to degrees
    dms_str: dtring specifying degrees minutes and seconds in the format deg:min:sec
    """
    str_splt = map(float, dms_str.split(':'))
    deg = np.abs(str_splt[0]) + str_splt[1]/60. + str_splt[2]/3600.
    if str_splt[0] < 0:
        multiply = -1
    else:
        multiply = 1
    return round(multiply * deg, 2)

def jd2lst(jd):
    """
    Calculates local sideral time in radians from right ascension and julian date
    jd : julian date     
    """ 
    j0 = ephem.julian_date(0)
    hera.date = jd - j0
    lst = hera.sidereal_time()
    return lst

def ralst2ha(ra, lst):
    """
    Calculates hour angle in radians given right ascension and lst in radians
    ra :  right ascension in radians
    lst : local sidereal time in radians
    """
    ha = lst - ra
    if ha > 2 * np.pi :  ha -= 2 * np.pi
    return ha

def hadec2azalt(dec, ha):
    """
    Calculates azimuth-altitude or horizontal coordinates from declination and hour anle in radians
    dec : declination in radians
    ha : hour anle in radians
    """
    sin_alt = np.sin(dec) * np.sin(hera.lat) + np.cos(dec) * np.cos(hera.lat) * np.cos(ha)
    alt = np.arcsin(sin_alt)
    cos_az = (np.sin(dec) - np.sin(alt) * np.sin(hera.lat)) / (np.cos(alt) * np.cos(hera.lat))
    az = np.arccos(cos_az)
    az = np.where(ha > 0, 2 * np.pi - az, az)
    return az, alt    

