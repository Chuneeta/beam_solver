import numpy as np
import nose.tools as nt
from beam_solver import coord_utils as ct

def test_deg2hms():
    ra_str = ct.deg2hms(0)
    nt.assert_equal(ra_str, '0h0m0.0s')
    ra_str = ct.deg2hms(30.01)
    nt.assert_equal(ra_str, '2h0m2.4s')

def test_negative_ra():
    nt.assert_raises(AssertionError, ct.deg2hms, -30.0)

def test_deg2dms():
    dec_str = ct.deg2dms(0)
    nt.assert_equal(dec_str, '0d0m0.0s')
    dec_str = ct.deg2dms(-30.01)
    nt.assert_equal(dec_str, '-30d0m36.0s')

def test_hms2deg():
    ra_d = ct.hms2deg('0:0:0.0')
    nt.assert_equal(ra_d, 0.0)
    ra_d = ct.hms2deg('2:0:2.4')
    nt.assert_equal(ra_d, 30.01)

def test_negative_hms():
    nt.assert_raises(AssertionError, ct.hms2deg, '-30:0:0')
    nt.assert_raises(AssertionError, ct.hms2deg, '30:-2:0')
    nt.assert_raises(AssertionError, ct.hms2deg, '30:0:-3.0')

def test_dms2deg():
    dec_d = ct.dms2deg('0:0:0')
    nt.assert_equal(dec_d, 0.0)
    dec_d = ct.dms2deg('-30:0:36.0')
    nt.assert_equal(dec_d, -30.01)

def test_jd2lst():
    lst0 = 3.7962537629153035 * np.pi/12.
    jd = 2458115.330633037
    lst = ct.jd2lst(jd)
    nt.assert_almost_equal(lst, lst0, 4)

def test_ralst2ha():
    lst = 3.7962537629153035 * np.pi/12.
    ra = 3.7962537629153035 * np.pi/12.
    ha = ct.ralst2ha(ra, lst)
    nt.assert_almost_equal(ha, 0.0)
    lst = 26.6 * np.pi/12.
    ra = 2.1 * np.pi/12.
    ha = ct.ralst2ha(ra, lst)
    nt.assert_almost_equal(ha, (lst - ra) - 2*np.pi, 4)

def test_hadec2azalt():
    dec = -30.721388888888885 * np.pi/180
    ha = 0.0
    az, alt = ct.hadec2azalt(dec, ha)
    nt.assert_almost_equal(az, np.pi/2, 5)
    nt.assert_almost_equal(alt, np.pi/2, 5)
    az, alt = ct.hadec2azalt(np.array([-30.721388888888885 * np.pi / 180, 90.0 * np.pi / 180]), np.array([0.0, 0.0]))
    np.testing.assert_almost_equal(az, np.array([np.pi/2, 0.0]))
    np.testing.assert_almost_equal(alt, np.array([np.pi/2, -30.721388888888885 * np.pi / 180]))
