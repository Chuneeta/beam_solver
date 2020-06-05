from collections import OrderedDict
import numpy as np
import hera_cal as hc
import pyuvdata
from beam_solver import get_redbls as gr
from hera_cal.io import HERAData, HERACal
from hera_cal.apply_cal import calibrate_in_place
from hera_cal.redcal import get_reds
import copy

# polarization mapping that are used in hera_cal
pols_dict = {'xx': 'ee',
             'xy': 'en',
             'yx': 'ne',
             'yy': 'nn'
}

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

    pol_hc = pols_dict[pol] # polarization convention used in heracal
    pos_dict = generate_antdict(uvd)
    red_bls = hc.redcal.get_reds(pos_dict, pols=[pol_hc])
    red = gr.RBL(red_bls)

    # selecting good antennas
    uvf = read_uvfits(uvfits)
    mod_bls = np.array(uvf.get_antpairs())

    omni_gains, omni_flags = hc.io.load_cal(omni_calfits)
    abs_gains, abs_flags = hc.io.load_cal(abs_calfits)

    res_data = copy.deepcopy(data)
    flag_data = np.ones((flag.shape), dtype=flag.dtype)
    for mbl in mod_bls:
        bl_grp = red[tuple(mbl) + (pol_hc,)]
        # somewhow the conjugate model visibilities are stored in the uvfits file
        for blp in bl_grp:
            bl = (blp[0], blp[1], pol) 
            mod_data = np.conj(uvf.get_data(tuple(mbl) + (pol,)).copy())
            inds = uvd.antpair2ind(bl[0], bl[1])
            _sh1, _sh2 = mod_data.shape
            try:
                mod_data *= omni_gains[bl[0], 'J{}'.format(pol_hc)] * np.conj(omni_gains[bl[1], 'J{}'.format(pol_hc)])
                mod_data /=  abs_gains[bl[0], 'J{}'.format(pol_hc)]  *  np.conj(abs_gains[bl[1], 'J{}'.format(pol_hc)])
                data_bl = uvd.get_data(bl)
                residual = data_bl - mod_data
                res_data[inds, :, :, :] = residual.reshape((_sh1, 1, _sh2, 1))
                flag_data[inds, :, :, :] = np.logical_or(uvf.get_flags(mbl).reshape(_sh1, 1, _sh2, 1), uvd.get_flags(bl).reshape(_sh1, 1, _sh2, 1))
            except KeyError:
                continue

    # writing data to UV file
    if outfile is None:
        outfile = uvfile + '.res'
    uvd_new = copy.deepcopy(uvd)
    uvd_new.data_array = res_data
    uvd_new.flag_array = flag_data
    uvd_new.write_miriad(outfile, clobber=clobber)

def generate_residual_IDR2_2(uvh5_file, omni_vis, omni_calfits, abs_calfits, outfile, clobber=False):
    # reading uvh5 data file
    hd = HERAData(uvh5_file)
    data, flags, nsamples = hd.read(polarizations=['ee', 'nn'])

    # reading omnical model visibilities
    hd_oc = HERAData(omni_vis)
    omnivis, omnivis_flags, _ = hd_oc.read()

    uvo = pyuvdata.UVData()
    uvo.read_uvh5(omni_vis)
    
    # reading calfits file
    hc = HERACal(omni_calfits)
    oc_gains, oc_flags, oc_quals, oc_total_quals = hc.read()

    hc = HERACal(abs_calfits)
    ac_gains, ac_flags, ac_quals, ac_total_quals = hc.read() 

    # calibrating the data
    abscal_data, abscal_flags = copy.deepcopy(data), copy.deepcopy(flags)
    calibrate_in_place(abscal_data, ac_gains, data_flags=abscal_flags, cal_flags=ac_flags)

    res_data, res_flags = copy.deepcopy(hd.data_array), copy.deepcopy(hd.flag_array) 
    resdata, resflags = copy.deepcopy(abscal_data), copy.deepcopy(abscal_flags)
    for i, p in enumerate(['ee', 'nn']):
        # reading omnical model visibilities
        hd_oc = HERAData(omni_vis)
        omnivis, omnivis_flags, _ = hd_oc.read(polarizations=[p])
        mod_bls = list(omnivis.keys())
        red_bls = get_reds(hd.antpos, pols=p)
        red = gr.RBL(red_bls)
        for mbl in mod_bls:
            bl_grp = red[tuple(mbl[0:2]) + ('J{}'.format(p),)]
            for blp in bl_grp:
                bl = (blp[0], blp[1], p)
                inds = hd.antpair2ind(bl)
                omnivis_scaled = omnivis[mbl] * oc_gains[(blp[0], 'J{}'.format(p))] * np.conj(oc_gains[(blp[1], 'J{}'.format(p))])
                omnivis_scaled /= (ac_gains[(blp[0], 'J{}'.format(p))] * np.conj(ac_gains[(blp[1], 'J{}'.format(p))]))
                resdata[bl] = abscal_data[bl] - omnivis_scaled
                resflags[bl] = abscal_flags[bl]
                res_data[inds, 0, : ,i] = resdata[bl]
                res_flags[inds, 0, :, i] = resflags[bl]

    # writing to file
    hd.data_array = res_data
    hd.flag_array = res_flags
    hd.write_uvh5(outfile, clobber=clobber)
    
