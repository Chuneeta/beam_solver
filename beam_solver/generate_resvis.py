from collections import OrderedDict
import numpy as np
import hera_cal as hc
import pyuvdata
from beam_solver import get_redbls as gr
import copy
import pylab

def read_uvfile(uvfile):
    """
    Reads in  miriad file and returns the observed visibilities
and the corresponding flags.

    Parameters
    ----------
    uvfile : string
        Name of input uvfile containing the visibilities and
        corresponding metadata.
    """
    uvd = pyuvdata.UVData()
    uvd.read_miriad(uvfile)
    return uvd

def read_calfits(calfits):
    """
    Reads in calfits file and returns the gain solutions and 
    the corresponding flags.

    Parameters
    ----------
    calfits : string
        Name of input calfits files containing the gain solutions
        and the corresponding metadata.
    """
    gains, gain_flags = hc.io.load_cal(calfits)
    return gains, gain_flags

def read_uvfits(uvfits):
    """
    Reads in uvfits file and returns the UVData object containing 
    the model visibilities and corresponding metadata.

    Parameters
    ----------
    uvfits : string
        Name of input uvfits files containing the model visibilities
        and the corresponding metadata.
    """
    uvf = pyuvdata.UVData()
    uvf.read_uvfits(uvfits)
    uvf.unphase_to_drift()
    return uvf

def generate_antdict(uvd):
    """
    Generates dictionary with antenna numbers as the keys and antenna positions
    as the corresponding items.
    NOTE : input to hera_cal.redcal.get_reds function
    
    Parameters
    ----------
    uvd : UVData object
            UVData object containing interferometric visibilities and the required
            metadata.
    """
    antpos = uvd.antenna_positions
    antnums = uvd.antenna_numbers
    pos_dict = OrderedDict()
    for ii, an in enumerate(antnums):
        pos_dict[an] = antpos[ii, :]
    return pos_dict

def generate_residual(uvfile, uvfits, omni_calfits, abs_calfits, pol, outfile=None, clobber=False):
    """
    Generate residual visibilities by subtracting the model visbilities
    obtained during omnical from the calibrated data

    Parameters
    ----------
    uvfile : string
        Name of input uvfile containing the visibilities and
        corresponding metadata.

    uvfits : string
        Name of input uvfits file containing the model visibilities and 
        necessary metadata.    

    old_calfits : string
        Calfits file containing gain solutions obtained via redundant 
        calibration (omnical).

    new_calfits : string
        Calfits file containing gain solutions obtained from absolute
        calibration combined with the gain solutions obtained from omnical.    

    pol : string
        Polarization, can be xx, xy , yx or yy

    outfile : string
        Name of output file name containing the residual visibilities.

    clobber : boolean
        If True, overwrites the existing file by the new one.
        Default is False
    """
    uvd = read_uvfile(uvfile)
    data = uvd.data_array
    flag = uvd.flag_array

    #aa = hc.utils.get_aa_from_uv(uvd)
    #info = hc.omni.aa_to_info(aa)
    pos_dict = generate_antdict(uvd)
    red_bls = hc.redcal.get_reds(pos_dict)
    #red_bls = np.array(info.get_reds())
    red = gr.RBL(red_bls)

    # selecting good antennas
    uvf = read_uvfits(uvfits)
    mod_bls = np.array(uvf.get_antpairs())

    omni_gains, omni_flags = hc.io.load_cal(omni_calfits)
    abs_gains, abs_flags = hc.io.load_cal(abs_calfits)

    res_data = np.ndarray((data.shape), dtype=data.dtype)
    flag_data = np.ndarray((flag.shape), dtype=flag.dtype)
    for mbl in mod_bls:
        bl_grp = red[tuple(mbl) + (pol,)]
        if len(bl_grp) > 1: 
            # somewhow the conjugate model visibilities are stored in the uvfits file
            for bl in bl_grp:
                mod_data = np.conj(uvf.get_data(tuple(mbl) + (pol,)).copy())
                inds = uvd.antpair2ind(bl[0], bl[1])
                _sh1, _sh2 = mod_data.shape
                try:
                    mod_data *= omni_gains[bl[0], 'J{}'.format(pol)] * np.conj(omni_gains[bl[1], 'J{}'.format(pol)])
                    mod_data /=  abs_gains[bl[0], 'J{}'.format(pol)]  *  np.conj(abs_gains[bl[1], 'J{}'.format(pol)])
                    mod_data *= ~uvd.get_flags(bl)
                    data_bl = uvd.get_data(bl) * ~uvd.get_flags(bl)
                    residual = data_bl - mod_data                    
                    res_data[inds, :, :, :] = residual.reshape((_sh1, 1, _sh2, 1))
                except KeyError:
                    flag_data[inds, :, :, :] = True
                    res_data[inds, :, :, :] = 0 + 0j
                    continue
        else:
            inds = uvd.antpair2ind(bl_grp[0][0], bl_grp[0][1])
            flag_data[inds, :, :, :] = True
            res_data[inds, :, :, :] = 0 + 0j

    # writing data to UV file
    if outfile is None:
        outfile = uvfile + '.res'
    uvd_new = copy.deepcopy(uvd)
    uvd_new.data_array = res_data
    uvd_new.flag_array = flag_data
    uvd_new.write_miriad(outfile, clobber=clobber)
