import numpy as np
import linsolve
import aipy.utils


class BeamSolving(object):
    
    def __init__(self, catalogue=None):
        """
        Object to store a catalogue or multiple astronomical sources to solve for the beam parameters
        
        Parameters
        ----------
        catalogue : string
            Txt file containing list of source
