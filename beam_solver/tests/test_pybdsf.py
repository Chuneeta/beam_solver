from beam_solver.data import DATA_PATH
from beam_solver import pybdsf as pb
import numpy as np
import nose.tools as nt
import os
import glob
import collections

fitsfile = os.path.join(DATA_PATH, '2458115.23736.xx.fits')

class Test_ProcessImage():
    def test_run_bdsf(self):
        bd = pb.ProcessImage([fitsfile])
        bd.run_bsdf(clobber=True)

    def test_read_gaul(self):
        bd = pb.ProcessImage([fitsfile])
        bd.run_bsdf(clobber=True)
        gaulfiles = glob.glob(DATA_PATH + '/*.gaul')
        bd.read_gaul(gaulfiles)
        nt.assert_true(isinstance(bd.srcdict, collections.OrderedDict))
        keys = bd.srcdict.keys()
        nt.assert_true('ra' in keys)
        nt.assert_true('dec' in keys)
        nt.assert_true('e_ra' in keys)
        nt.assert_true('e_dec' in keys)
        nt.assert_true('tflux' in keys)
        nt.assert_true('pflux' in keys)
        nt.assert_true('e_tflux' in keys)
        nt.assert_true('e_pflux' in keys)
        nt.assert_equal(len(bd.srcdict['ra']), len(bd.srcdict['dec'])) 

# removing gaul and log files
os.system('rm -r {}/*.gaul'.format(DATA_PATH))
os.system('rm -r {}/*.log'.format(DATA_PATH))
