from collections import OrderedDict
import numpy as np
import os, sys
import bdsf
import copy
import pickle

bsdf_npar = 47 # number of parameters returned by PyBDSF for each source

class ProcessImage(object):
    def __init__(self, fitsfiles):
        """
        Object to store the fitsfiles on which PyBDSF will be executed and
        the resulting catalog will be stored.
        Parameters
        ----------
        fitsfiles : numpy.ndarray or list
            Numpy ndarray of list of strings representing the path to the fitsfiles.
        """    
        self.fitsfiles = fitsfiles
        self.srcdict = OrderedDict()
        self.uniq_ind = None

    def _process_image(self, fitsfile, thresh_isl, thresh_pix):
        """
        Executes PyBDSF on the fitsfile
        Parameters
        ----------
        fitsfile : string
            Name of input fitsfile
        thresh_isl : float
            Threshold for the island boundary in number of sigma above the mean. 
            Determines extent of island used for fitting.
        thresh_pix : float
            Source detection threshold: threshold for the island peak in number 
            of sigma above the mean.
        """
        bdsf_dict = {'rms_map': None,
                     'thresh' : 'hard',
                     'thresh_isl': thresh_isl,
                     'thresh_pix': thresh_pix
                    }
        return bdsf.process_image(fitsfile, **bdsf_dict)

    def run_bsdf(self, thresh_isl=5.0, thresh_pix=7.0, clobber=False):
        """
        Executes PyBDSF on multiples fitsfiles
        Parameters
        ----------
        thresh_isl : float
            Threshold for the island boundary in number of sigma above the mean.
            Determines extent of island used for fitting.
        thresh_pix : float
            Source detection threshold: threshold for the island peak in number
            of sigma above the mean.
        clobber : boolean
            If True overwrites any existing file. Default is True.
        """
        for fn in self.fitsfiles:
            img = self._process_image(fn, thresh_isl, thresh_pix)
            outfile = fn.replace('.fits', '.gaul')
            img.write_catalog(outfile=outfile, format='ascii', catalog_type='gaul', clobber=clobber)

    def read_gaul(self, gaulfiles):
        """
        Reads in the gaulfiles that PyBDSM output and stores the source parameters
        into a dictionary.
        Parameters
        ----------
        gaulfiles : list ot numpy.ndarray
            List of numpy.ndarray of strings representing the gaulfiles that PyBDSF 
            generates.
        """
        self.srcdict['ra'] = np.array([]); self.srcdict['e_ra'] = np.array([])
        self.srcdict['dec'] = np.array([]); self.srcdict['e_dec'] = np.array([])
        self.srcdict['tflux'] = np.array([]);  self.srcdict['e_tflux'] = np.array([])
        self.srcdict['pflux'] = np.array([]); self.srcdict['e_pflux'] = np.array([])
        for gaul in gaulfiles:
            gaularray = np.loadtxt(gaul, dtype='str')
            nsrcs = len(gaularray)
            if gaularray.ndim == 1:
                gaularray.shape = (1, bdsf_npar)
            self.srcdict.update({'ra': np.append(self.srcdict['ra'], map(float, gaularray[:, 4]))})
            self.srcdict.update({'e_ra': np.append(self.srcdict['e_ra'], map(float, gaularray[:, 5]))})
            self.srcdict.update({'dec': np.append(self.srcdict['dec'], map(float, gaularray[:, 6]))})
            self.srcdict.update({'e_dec': np.append(self.srcdict['e_dec'], map(float, gaularray[:, 7]))})
            self.srcdict.update({'tflux': np.append(self.srcdict['tflux'], map(float, gaularray[:, 8]))})
            self.srcdict.update({'e_tflux': np.append(self.srcdict['e_tflux'], map(float, gaularray[:, 9]))})
            self.srcdict.update({'pflux': np.append(self.srcdict['pflux'], map(float, gaularray[:, 10]))})    
            self.srcdict.update({'e_pflux': np.append(self.srcdict['e_pflux'], map(float, gaularray[:, 11]))})

    def save_to(self, outfile):
        """
        Saving the dictionary to pkl file
        Parameters
        ----------
        outfile : string    
            Name of the output file
        """
        outfile += '.pkl'
        pickle.dump(self.srcdict, open(outfile, 'wb'))
