from setuptools import setup
import sys
import os
from beam_solver import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('beam_solver', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('beam_solver', 'data') 

setup_args = {
    'name':         'beam_solver',
    'author':       'HERA Team',
    'url':          'https://github.com/Chuneeta/beam_solver',
    'license':      'BSD',
    'version':      version.version,
    'description':  'HERA Primary Beam Estimator.',
    'packages':     ['beam_solver'],
    'package_dir':  {'beam_solver': 'beam_solver'},
    'package_data': {'beam_solver': data_files},
    'install_requires': ['numpy>=1.14', 'scipy', 'matplotlib>=2.2'],
    'include_package_data': True,
    #'scripts': ['scripts/pspec_run.py', 'scripts/pspec_red.py',
    #            'scripts/bootstrap_run.py'],
    'zip_safe':     False,
}

if __name__ == '__main__':
    apply(setup, (), setup_args)
