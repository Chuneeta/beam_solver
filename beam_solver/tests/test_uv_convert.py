from beam_solver.data import DATA_PATH
from beam_solver import uv_convert as uc
import nose.tools as nt
import glob
import os

uvfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA') 
uvfits = uvfile + '.uvfits'
msfile = uvfile + '.ms'

class Test_UVConvert():
    def test_uv_convert(self):
        uvc = uc.UVConvert(uvfile)
        uvc.convert_uv(clobber=True)
        nt.assert_true(os.path.exists(uvfits))
        nt.assert_true(os.path.exists(msfile))
    
    def test_del_uvfits(self):
        uvc = uc.UVConvert(uvfile)
        uvc.convert_uv(del_uvfits=True, clobber=True)
        nt.assert_false(os.path.exists(uvfits))
 
    def test_delete_log(self):
        uvc = uc.UVConvert(uvfile)
        uvc.convert_uv(clobber=True)
        uvc.delete_log()
        cwd = os.getcwd()
        logfiles = glob.glob(cwd + '/*.log')
        lastfiles = glob.glob(cwd + '/*.last')
        nt.assert_equal(len(logfiles), 0)
        nt.assert_equal(len(lastfiles), 0)

        # deleting created msfile and uvfits file
        os.system('rm -rf {}'.format(uvfits))
        os.system('rm -rf {}'.format(msfile))
