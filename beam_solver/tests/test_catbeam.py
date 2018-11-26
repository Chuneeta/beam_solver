import unittest
import numpy as np
import nose.tools as nt
import os
import sys
from beam_solver import catbeam
from beam_solver.data import DATA_PATH
import pyuvdata
import copy

cstbeam1 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_151MHz.txt')
cstbeam2 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_152MHz.txt')
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')


# creating beam class
class Beam(object):
    def __init__(self):
        self.data = np.ones((128,))
        self.nsize = 128
        self.freq = 150e6


class Test_catBeamGauss():
    def test_generate_beam(self):
        # defining the variables
        mu = 15.
        sigma = 3.
        size = (31, 31)

        # generating gaussian beam
        beam = catbeam.catBeamGauss()
        beam.generate_beam(mu, sigma, size)
        size = 31
        beam.generate_beam(mu, sigma, size)

        # checking if the inputs are converted to floats
        beam = catbeam.catBeamGauss()
        beam.generate_beam(int(mu), int(sigma), size)

        # checking if it accepts only 2-dimensional input
        size = (31, 31, 31)
        beam = catbeam.catBeamGauss()
        nt.assert_raises(ValueError, beam.generate_beam, mu, sigma, size)


class Test_catBeamFits():
    def test_generate_beam(self):
        # generating beam at 150 MHz from beamfits file
        beam = catbeam.catBeamFits()
        beam.generate_beam(beamfits, 150e6, pol=['xx'])
        beam.generate_beam(beamfits, 150e6, pol='xx')
        beam.generate_beam(beamfits, 150.5e6, pol=['xx'])

        # testing beam objects
        uvb = pyuvdata.UVBeam()
        uvb.read_beamfits(beamfits)
        beam = catbeam.catBeamFits()
        beam.generate_beam(uvb, 150e6, pol='xx')
        uvb = Beam()
        nt.assert_raises(ValueError, beam.generate_beam, uvb, 150e6, pol='xx')

        # checking for frequency inputs
        beam = catbeam.catBeamFits()
        nt.assert_raises(ValueError, beam.generate_beam, beamfits, None, pol=['xx'])
        nt.assert_raises(ValueError, beam.generate_beam, beamfits, 50e6, pol=['xx'])

        # checking polarizations
        beam = catbeam.catBeamFits()
        beam.generate_beam(beamfits, 150e6, pol=['xx', 'yy'])
        nt.assert_raises(NotImplementedError, beam.generate_beam, beamfits, 150e6, pol=['xy'])


class Test_catBeamCst():
    def test_generate_beam(self):
        # testing files input and frequency range
        beam = catbeam.catBeamCst()
        beam.generate_beam(cstbeam1, 151e6, 151e6, pol=['xx'])
        beam.generate_beam([cstbeam1, cstbeam2], [151e6, 152e6], 151.5e6, pol=['xx'])
        nt.assert_raises(AssertionError, beam.generate_beam, [], 151e6, 140e6, pol=['xx'])
        nt.assert_raises(ValueError, beam.generate_beam, [], [], 140e6, pol=['xx'])
        nt.assert_raises(ValueError, beam.generate_beam, cstbeam1, 151e6, 140e6, pol=['xx'])
        nt.assert_raises(ValueError, beam.generate_beam, cstbeam1, 151e6, None, pol=['xx'])
        nt.assert_raises(ValueError, beam.generate_beam, [cstbeam1, cstbeam2], [151e6, 152e6], 140e6, pol=['xx'])

        # testing polarizations
        beam.generate_beam(cstbeam1, 151e6, 151e6, pol=['xx', 'yy'])
        beam.generate_beam([cstbeam1, cstbeam2], [151e6, 152e6], 151.5e6, pol=['xx', 'yy'])


if __name__ == "__main__":
    unittest.main()
