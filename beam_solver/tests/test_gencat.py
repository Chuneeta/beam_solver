import numpy as np
from beam_solver import gencat as gc
from beam_solver.data import DATA_PATH
import copy
import os
import sys
import nose.tools as nt

# fitsfiles
DATA_PATH = '/Users/Ridhima/Documents/ucb_projects/beam_characterization/beam_solver/beam_solver/data'
fitsfile1_xx = os.path.join(DATA_PATH, '2458115.23736.xx.fits')
fitsfile2_xx = os.path.join(DATA_PATH, '2458115.24482.xx.fits')
altered_xx = os.path.join(DATA_PATH, '2458115.24482.xx.altered.fits')
fitsfiles_xx = [fitsfile1_xx, fitsfile2_xx]
fitsfile1_yy = os.path.join(DATA_PATH, '2458115.23736.yy.fits')
fitsfile2_yy = os.path.join(DATA_PATH, '2458115.24482.yy.fits')
fitsfiles_yy = [fitsfile1_yy, fitsfile2_yy]

# right ascension and declination values
ras = [30.01713089, 27.72922349, 36.75248962, 34.2415497, 78.3776346, 74.03785837]
decs = [-30.88211818, -29.53377208, -30.63958257, -29.93990039, -30.48595805, -30.08651873]


def test_get_unique():
    gc.get_unique(ras=ras, decs=decs)
    ras1 = copy.deepcopy(ras)
    decs1 = copy.deepcopy(decs)
    ras1[1] = ras[0]
    decs1[1] = decs[0]
    gc.get_unique(ras=ras1, decs=decs1)

    # checking for Value Errors
    nt.assert_raises(ValueError, gc.get_unique, ras=[], decs=decs)
    nt.assert_raises(ValueError, gc.get_unique, ras=ras, decs=[])
    nt.assert_raises(ValueError, gc.get_unique, ras=[], decs=[])

    # checking for assertions
    nt.assert_raises(AssertionError, gc.get_unique, ras=ras, decs=decs[0:4])


class Test_genCatBase():
    def test_gen_catalog(self):
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras, decs=decs)
        cat.gen_catalog()
        cat = gc.genCatBase(fits=fitsfiles_yy, ras=ras, decs=decs)
        cat.gen_catalog()
        cat = gc.genCatBase(fits=fitsfiles_xx[0], ras=ras, decs=decs)
        cat.gen_catalog()
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras[0], decs=decs[0])
        cat.gen_catalog()
        cat = gc.genCatBase(fits=np.array(fitsfiles_xx), ras=np.array(ras), decs=np.array(decs))
        cat.gen_catalog()

        # checking for proper inputs
        ras1 = '34.5'
        decs1 = '-30.56'
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)
        cat = gc.genCatBase(fits=[30, 40], ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)
        ras1 = copy.deepcopy(ras)
        decs1 = copy.deepcopy(decs)
        ras1[0] = '34.5'
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)
        decs1[0] = '-30.56'
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)

        # checking for right ascension and declination range
        ras1[0] = -34.456
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras1, decs=decs)
        nt.assert_raises(ValueError, cat.gen_catalog)
        decs1[0] = 45.56
        cat = gc.genCatBase(fits=fitsfiles_xx, ras=ras, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)


class Test_genCatCross():
    def test_gen_catalog(self):
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras, decs=decs)
        cat.gen_catalog()
        cat = gc.genCatCross(fitsxx=fitsfiles_xx[0], fitsyy=fitsfiles_yy[0], ras=ras, decs=decs)
        cat.gen_catalog()
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras[0], decs=decs[0])
        cat.gen_catalog()

        # checking for proper inputs
        ras1 = '34.5'
        decs1 = '-30.56'
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)
        cat = gc.genCatCross(fitsxx=[30, 40], fitsyy=fitsfiles_yy, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)
        ras1 = copy.deepcopy(ras)
        decs1 = copy.deepcopy(decs)
        ras1[0] = '34.5'
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)
        decs1[0] = '-30.56'
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras1, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)

        # checking for right ascension and declination range
        ras1 = copy.deepcopy(ras)
        ras1[0] = -34.456
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras1, decs=decs)
        nt.assert_raises(ValueError, cat.gen_catalog)
        decs1 = copy.deepcopy(decs)
        decs1[0] = 45.56
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy, ras=ras, decs=decs1)
        nt.assert_raises(ValueError, cat.gen_catalog)

        # checking if fitsfile for xx and yy polarization are consistent
        cat = gc.genCatCross(fitsxx=fitsfiles_xx, fitsyy=fitsfiles_yy[0:1], ras=ras1, decs=decs)
        nt.assert_raises(AssertionError, cat.gen_catalog)

        # checking if julian dates for xx and yy polarizations are consistent
        cat = gc.genCatCross(fitsxx=fitsfiles_xx[0:1], fitsyy=fitsfiles_yy[1:2], ras=ras1, decs=decs)
        nt.assert_raises(ValueError, cat.gen_catalog)

        # checking if frequencies for xx an yy polarizations are consistent
        cat = gc.genCatCross(fitsxx=[altered_xx], fitsyy=fitsfiles_yy[1:2], ras=ras1, decs=decs)
        nt.assert_raises(ValueError, cat.gen_catalog)


if __name__ == "__main__":
    unittest.main()
