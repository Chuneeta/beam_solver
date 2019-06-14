import numpy as np
from beam_solver import generate_resvis as gr
from beam_solver.data import DATA_PATH
import os, sys
import nose.tools as nt
import pyuvdata
import collections

def test_read_uvfile():
    uvfile = os.path.join(DATA_PATH,  'zen.2458098.44615.xx.HH.uvc')
    uvd = gr.read_uvfile(uvfile)
    nt.assert_true(isinstance(uvd, pyuvdata.UVData))

def test_read_calfits():
    calfits = os.path.join(DATA_PATH,  'zen.2458098.44615.HH.uv.com.calfits')
    gains, gain_flags = gr.read_calfits(calfits)
    nt.assert_true(isinstance(gains, collections.OrderedDict))
    nt.assert_true(gains.keys(), 2)

def test_gain_flags():
    calfits = os.path.join(DATA_PATH,  'zen.2458098.44615.HH.uv.com.calfits')
    gains, gain_flags = gr.read_calfits(calfits)
    nt.assert_true(isinstance(gain_flags, collections.OrderedDict))
    nt.assert_true(gain_flags.keys(), 2)
    keys = list(gain_flags.keys())
    nt.assert_true(gain_flags[keys[0]].dtype, 'bool')

def test_read_uvfits():
    uvfits = os.path.join(DATA_PATH, 'zen.2458098.44615.HH.uv.com.uvfits')
    uvf = gr.read_uvfits(uvfits)
    nt.assert_true(isinstance(uvf, pyuvdata.UVData))

def test_generate_antdict():
    uvfile = os.path.join(DATA_PATH,  'zen.2458098.44615.xx.HH.uvc')
    uvd = gr.read_uvfile(uvfile)
    pos_dict = gr.generate_antdict(uvd)
    nt.assert_true(isinstance(pos_dict, collections.OrderedDict))
    uvd = gr.read_uvfile(uvfile)
    nt.assert_true(len(pos_dict.keys()), len(uvd.antenna_positions))

def test_generate_residual():
    uvfile = os.path.join(DATA_PATH,  'zen.2458098.44615.xx.HH.uvc')
    omni_calfits = os.path.join(DATA_PATH,  'zen.2458098.44615.HH.uv.com.calfits')
    abs_calfits = os.path.join(DATA_PATH,  'zen.2458098.44615.HH.xx.uv.abs.com.calfits')
    uvfits = os.path.join(DATA_PATH, 'zen.2458098.44615.HH.uv.com.uvfits')
    outfile = '/Users/Ridhima/Documents/ucb_projects/beam_characterization/beam_solver/beam_solver/data/resfile.uv'
    gr.generate_residual(uvfile, uvfits, omni_calfits, abs_calfits, 'xx', outfile=outfile, clobber=True)
    nt.assert_true(os.path.exists(outfile))
    os.system('rm -rf {}'.format(outfile))
