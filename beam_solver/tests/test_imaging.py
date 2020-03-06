from beam_solver.data import DATA_PATH
from beam_solver import imaging as im
from beam_solver import uv_convert as uc
import nose.tools as nt
import os, sys
import glob

uvfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
uvfits = uvfile + '.uvfits'
msfile = uvfile + '.ms'
imagename = os.path.join(DATA_PATH, '2457698.40355.xx')
fitsname = imagename + '.fits'

uvc = uc.UVConvert(uvfile)
uvc.convert_uv(clobber=True)

src_dict = {1: ('1:55:33.98', '-28:37:32.70')}

class Test_Imaging():
    def test_generate_image(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        nt.assert_true(os.path.exists(imagename + '.image'))
        os.system('rm -rf {}*'.format(imagename))

    def test_to_fits(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        img.to_fits(imagename + '.image', overwrite=True)
        nt.assert_true(os.path.exists(imagename + '.fits'))
        os.system('rm -rf {}*'.format(imagename))

    def test_remove_image(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        img.remove_image(imagename)
        nt.assert_false(os.path.exists(imagename + '.model'))
        nt.assert_true(os.path.exists(imagename + '.image'))
        os.system('rm -rf {}*'.format(imagename))
 
    def test_remove_all(self):
        img = im.Imaging(msfile)
        img.generate_image(imagename)
        img.remove_image(imagename, del_img=True)
        nt.assert_false(os.path.exists(imagename + '.image'))
        os.system('rm -rf {}*'.format(imagename))

    def test_subtract_model(self):
        img = im.Imaging(msfile)
        outfile = 'residual_vis.ms'
        img.subtract_model(outfile, del_script=True)
        nt.assert_true(os.path.exists(outfile))    
        os.system('rm -rf {}'.format(outfile))

    def test_delete_log(self):
        img = im.Imaging(msfile)
        img.delete_log()
        cwd = os.getcwd()
        logfiles = glob.glob(cwd + '/*.log')
        lastfiles = glob.glob(cwd + '/*.last')
        nt.assert_equal(len(logfiles), 0)
        nt.assert_equal(len(lastfiles), 0)

class Test_Subtract():
    def test_make_image(self):
        sub = im.Subtract(msfile)
        sub.make_image(imagename, fitsname, overwrite=True)
        nt.assert_true(os.path.exists(fitsname))
        nt.assert_false(os.path.exists(imagename + '.image'))

    def test_srcdict_to_list(self):
        sub = im.Subtract(msfile, src_dict)
        ra, dec = sub.srcdict_to_list()
        nt.assert_true(ra, 28.89)
        nt.assert_true(dec, -28.63)

    def test_extract_flux(self):
        sub = im.Subtract(msfile, src_dict)
        sub.make_image(imagename, fitsname, overwrite=True)
        ra, dec = sub.srcdict_to_list()
        flux = sub.extract_flux(fitsname, ra, dec)
        nt.assert_true(flux[0], 0.00684369)

    def test_get_freq(self):
        sub = im.Subtract(msfile, src_dict)
        sub.make_image(imagename, fitsname, overwrite=True)
        freq = sub.get_freq(fitsname)
        nt.assert_true(freq, 153.7109429128)

    def test_subtract_model(self):
        newmsfile = msfile.replace('.ms', '.copy.ms')
        os.system('cp -rf {} {}'.format(msfile, newmsfile))
        sub = im.Subtract(newmsfile, src_dict)
        sub.subtract_model(imagename, fitsname=fitsname)
        nt.assert_true(os.path.exists(fitsname))
