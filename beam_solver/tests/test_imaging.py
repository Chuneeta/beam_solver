from beam_solver.data import DATA_PATH
from beam_solver import imaging as im
from beam_solver import uv_convert as uc
import nose.tools as nt
import os
import glob

uvfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
uvfits = uvfile + '.uvfits'
msfile = uvfile + '.ms'
imagename = os.path.join(DATA_PATH, '2457698.40355.xx')
fitsname = imagename + '.fits'

uvc = uc.UVConvert(uvfile)
uvc.convert_uv(clobber=True)

class Test_Imaging():
    def test_generate_image(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        nt.assert_true(os.path.exists(imagename + '.image'))

    def test_to_fits(self):
        img = im.Imaging(msfile)
        img.to_fits(imagename)
        nt.assert_true(os.path.exists(imagename + '.fits'))

    def test_remove_image(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        img.remove_image(imagename)
        nt.assert_false(os.path.exists(imagename + '.model'))
        nt.assert_true(os.path.exists(imagename + '.image'))
 
    def test_remove_all(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        img.remove_image(imagename, del_img=True)
        nt.assert_false(os.path.exists(imagename + '.image'))

    def test_delete_log(self):
        img = im.Imaging(msfile)
        img.delete_log()
        cwd = os.getcwd()
        logfiles = glob.glob(cwd + '/*.log')
        lastfiles = glob.glob(cwd + '/*.last')
        nt.assert_equal(len(logfiles), 0)
        nt.assert_equal(len(lastfiles), 0)
