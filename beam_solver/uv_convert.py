from astropy.time import Time
import casa_utils as ct
import pyuvdata
import os

class UVConvert(object):
    def __init__(self, uvfile, outfile=None):
        """
        Object to store measurement sets of Miriad files containing visibilities in order to
        convert them to measurements.
        Parameters
        ----------
        uvfile : str
            Input miriad file containing visibilities are required metadata.
        outfile : str
            Output name of the measurement set file
        """
        self.uvfile = uvfile
        if outfile is None:
            self.outfile = '{}.ms'.format(self.uvfile)
        else:
            self.outfile = outfile

    def convert_uv(self, phs=None, del_uvfits=False, script='uvfits2ms', del_script=True, clobber=False):
        """
        Converts Miriad file to Measurement set (MS)
        Parameters
        ----------
        phs : float, optional
            Julian date at which to phase the visibilities. By default the visibilities are phased to middle timestamp of the file
        del_uvfits : boolean, optional
            If True, deleted the uvfits file that is created during the conversion from uvfile to ms.
            Default is False.
        script : string, optional
            Casa script created on-the-fly to execute the casa task.
            Default is uvfits2ms.
        del_script: boolean, optional
            If True, deletes the on-fly created casa script. Default is True.
        clobber : boolean, optional
            If True, overwrites the existing file by the new one.
            Default is False    
        """
        uvd = pyuvdata.UVData()
        uvd.read_miriad(self.uvfile)
        times = uvd.time_array
        phs_time = times[int(len(times)/2.)] if phs is None else phs

        print ('Phasing visibilities to {}'.format(phs_time))
        uvd.phase_to_time(Time(phs_time, format='jd', scale='utc'))
        # converting to uvfits
        uvfits = self.uvfile + '.uvfits'
        print ('Converting {} to {}'.format(self.uvfile, uvfits))
        uvd.write_uvfits(uvfits, spoof_nonessential=True)
        # converting to mset
        if clobber:
            if os.path.exists(self.outfile):
                os.system('rm -rf {}'.format(self.outfile))
        ct.uvfits2ms(uvfits, outfile=self.outfile, script=script, delete=del_script)
        # removing uvfits
        if del_uvfits:
            os.system('rm -rf {}'.format(uvfits))
        # removing log files
        os.system('rm -rf *.log')
    
    def delete_log(self):
        """
        Deletes unecessary log files created during execution
        """
        os.system('rm -rf *.log')
        os.system('rm -rf *.log~')
        os.system('rm -rf *.last')
