import numpy as np
import ephem

# setting HERA observer
hera_lat = '-30:43:17'
hera_lon = '21:25:40.08'
hera = ephem.Observer()
hera.lat, hera.long, hera.elevation = hera_lat, hera_lon, 0.0
j0 = ephem.julian_date(0)

def deg2hms(ra):
    """
    Converts degrees to hours minutes seconds

    Parameters
    ----------
    ra : float
        Right ascension or any value to convert to hours minutes seconds
    """

    ra_hrs = ra / 15.
    assert ra_hrs < 0, "right ascension or value should be positive"
    ra_h = int(ra_hrs)
    ra_mins = (ra_hrs - ra_h) * 60
    ra_m = int(ra_mins)
    ra_secs = (ra_mins - ra_m) * 60
    ra_s = round(ra_secs, 2) 
     
    return '{}h{}m{}s'.format(ra_h, ra_m, ra_s)

def deg2dms(dec):
    """
    Converts to degrees to degrees minutes seconds

    Parameters
    ----------
    dec : float
        Declination or any other value to convert to degrees minutes seconds
    """

    dec_a = np.abs(dec)
    dec_deg = int(dec_a)
    dec_mins =  (dec_a - dec_deg) * 60
    dec_m = int(dec_mins)
    dec_secs = (dec_mins - dec_m) * 60
    dec_s = round(dec_secs, 2)

    if dec < 0 :
        return '{}d{}m{}s'.format(-1 * dec_deg, dec_m, dec_s)
    else:
        return '{}d{}m{}s'.format(dec_deg, dec_m, dec_s)

def hms2deg(hms_str):
    """
    Converts hours minutes seconds to degrees
    
    Parameters
    ----------
    hms_str : string
        String specifying hours minutes seconds in the format hrs:min:sec
    """

    str_splt = map(float, hms_str.split(':'))
    hrs = str_splt[0] + str_splt[1]/60. + str_splt[2]/3600. 
    deg = hrs * 15

    return deg

def dms2deg(dms_str):
    """
    Converts degrees minutes seconds to degrees

    Parameters
    ----------
    dms_str: string
        String specifying degrees minutes and seconds in the format deg:min:sec

    """
    str_splt = dms_str.split(':')
    deg = np.abs(str_splt[0]) + str_splt[1]/60. + str_splt[2]/3600.
    if str_splt[0] < 0:
        multiply = -1
    else:
        multiply = 1

    return multiply * deg

def rajd2ha(ra, jd):
    """
    Calcualtes hour angle in degrees from right ascension and julian date

    Parameters
    ---------
    ra : float
        Right ascension in degrees
    
    jd : float
        Julian date     
    """
    
    ra_r = ra * np.pi/180 # convert to radians
    j0 = ephem.julian_date(0)
    hera.date = jd - j0
    lst = hera.sidereal_time()
    ha = lst - ra_r

    return lst * 180/np.pi, ha * 180/np.pi

def hadec2azalt(ha, dec, lat):
    """
    Calculates azimuth-altitude from hour angle and declination

    Parameters
    ----------
    ha : float
        Hour angle in degrees
    
    dec : float
        Declination in degrees

    lat : float
        Latitude of observer in degrees

    Returns
    -------
    az, alt in degrees
    """
    
    cos_h = np.cos(ha * np.pi/180)
    sin_h = np.sin(ha * np.pi/180)
    cos_d = np.cos(dec * np.pi/180)
    sin_d = np.sin(dec * np.pi/180)
    cos_l = np.cos(lat * np.pi/180)
    sin_l = np.sin(lat * np.pi/180)

    x = -1 * cos_h * cos_d * sin_l + sin_d * cos_l
    y = -1 * sin_h * cos_d  
    z = cos_h * cos_d * cos_l + sin_d * sin_l
    r = np.sqrt(x**2 + y**2)

    az = np.arctan2(y , x) * 180/np.pi
    alt = np.arctan2(z , r) * 180/np.pi
    if az < 0: az += 360.
    
    return az, alt    

def radec2azalt(jd, ra, dec):
    """
    """
        
    hera.date = jd - j0 
    src=ephem.FixedBody()
    src._ra = ra * np.pi/180.
    src._dec = dec * np.pi/180.
    src.compute(hera)  
    az = src.az
    alt = src.alt

    return az * 180/np.pi, alt * 180/np.pi  
