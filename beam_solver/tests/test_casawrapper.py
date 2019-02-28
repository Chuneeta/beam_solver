import nose.tools as nt
from beam_solver import casawrapper as cw
import os

direc = os.getcwd()
scriptname = os.path.join(direc, 'execute.py')

def test_create_casa_options():
    kwargs = cw.create_casa_options(nologger='0', nogui='0', nologfile='0')
    nt.assert_equal(kwargs, {'nologger': '0', 'nogui': '0', 'nologfile': '0'})

def test_call_casa_task():
    casa_opts = cw.create_casa_options(nologger='0', nogui='0', nologfile='0')
    cw.call_casa_task(task="importuvfits", script="execute", task_options={'vis':"'test.ms'"}, casa_options=casa_opts, delete=False, verbose=True)
    nt.assert_true(os.path.exists(scriptname))
    fl = open(scriptname)
    lines = []
    for line in fl.readlines():
        lines.append(line)
    answer = ['default(importuvfits)\n', "vis='test.ms'\n", 'go()']
    nt.assert_equal(lines, answer)

def test_call_casa_script():
    direc = os.getcwd()
    scriptname = os.path.join(direc, 'execute.py')
    cw.call_casa_script(scriptname, casa_opts={'nologger': '0'}, delete=True)
    nt.assert_false(os.path.exists(scriptname))

# removing script
os.system('rm -r *.log')
os.system('rm -r *.log~')
os.system('rm -rf {}'.format(scriptname))
