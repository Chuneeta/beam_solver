import os

def create_casa_options(**kwargs):
    """
    Returns input CASA options    
    """
    return kwargs

def call_casa_task(task, script, task_options={}, casa_options={}, delete=True, verbose=True):
    """
    Calls the specified CASA task
    Parameters
    ----------
    task: string 
        CASA task to execute, for example, clean, importuvfits, flagdata, etc.
    script: string
        Name of CASA script that will be created and executed
    task_options: dict
        Dictionary of the options and corresponding parameters for the CASA task.
        For example, for the clean task, task_options={'vis':msname, 'imagename':image, etc}
    casa_options: dict
        Dictionary of the CASA startup options.
        For example, casa_options={'maclogger':'0', 'nologger':'0'}    
    delete: boolean
        Deletes CASA script after execution
    verbose: boolean
        Displays/prints out the command being executed
    """
    # creating casa script
    scriptname = '{}.py'.format(script)
    if os.path.exists(scriptname):
        os.system('rm -r {}'.format(scriptname))
    stdout = open(scriptname, 'wb')
    line = 'default({})\n'.format(task)
    for key in task_options.keys():
        line += '{}={}\n'.format(key, task_options[key])
    line += 'go()'
    stdout.write(line)
    stdout.close()
    # casa otpions
    casa_out = ''	
    keys = casa_options.keys()
    # location of startup file
    if 'rcdir' in keys:
        casa_out += '--rcdir {} '.format(casa_options['rcdir'])
    # path to logfile
    if 'logfile' in keys:
        casa_out += '--logfile {} '.format(casa_options['logfile'])
    # logger to use on Apple systems
    if 'maclogger' in keys:
        casa_out += '--maclogger '
    # direct output to terminal
    if 'log2term' in keys:
        casa_out += '--log2term '
    # do not start CASA logger
    if 'nologger' in keys:
        casa_out += '--nologger '
    # do not creat log file 
    if 'nologfile' in keys:
        casa_out += '--nologfile '
    # avoid starting GUI tools
    if 'nogui' in keys:
        casa_out += '--nogui'
    # prompt color [NoColor, Linux, LightBG]
    if 'colors' in keys:
        casa_out += '--colors {} '.format(casa_options['colors'])
    # do not create ipython log
    if 'noipython' in keys:
        casa_out += '--noipython '
    command = 'casa {} -c {}'.format(casa_out, scriptname)
    if verbose:
        print ('CMD: {}'.format(command))
    os.system(command)
    if delete:  
        os.system('rm -r {}'.format(scriptname))        

def call_casa_script(scriptname, casa_opts='', delete=False, verbose=True):
    """
    Executing CASA script
    Parameters
    ----------
    scriptname : string
        CASA script to be executed
    casa_options: dict
        Dictionary of the CASA startup options.
        For example, casa_options={'maclogger':'0', 'nologger':'0'}
    delete: boolean
        Deletes CASA script after execution
    verbose: boolean
        Displays/prints out the command being executed
    """
    call_opts = ''
    for key in casa_opts:
        if casa_opts[key] == '0': call_opts += '--%s '%(key)    
    command = 'casa {} -c {}'.format(call_opts, scriptname)
    if verbose:
        print ("CMD:: ",command)
    os.system(command)
    if delete:
        os.system("rm -r " + scriptname)
