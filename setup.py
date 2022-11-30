from setuptools import setup, find_packages
import os
import re


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


def get_version():
    VERSIONFILE = os.path.join('src', 'sdynpy', '__init__.py')
    with open(VERSIONFILE, 'rt') as f:
        lines = f.readlines()
    vgx = '^__version__ = \"[0-9+.0-9+.0-9+]*[a-zA-Z0-9]*\"'
    for line in lines:
        mo = re.search(vgx, line, re.M)
        if mo:
            return mo.group().split('"')[1]
    raise RuntimeError('Unable to find version in %s.' % (VERSIONFILE,))

def get_ui_files(directory):
    paths = []
    for (path,directories,filenames) in os.walk(directory):
        for filename in filenames:
            if filename[-3:] == '.ui':
                paths.append(os.path.join(path,filename))
    return paths

setup(
    name='sdynpy',
    version=get_version(),
    description=('A Structural Dynamics Library for Python'),
    long_description=read("README.rst"),
    url='https://cee-gitlab.sandia.gov/structMechTools/structural-dynamics-python-libraries',
    author='Daniel P. Rohe',
    author_email='dprohe@sandia.gov',
    license='Sandia Proprietary',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    zip_safe=False,
    package_data={'': get_ui_files('src/sdynpy')},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "PyQt5",
        "pyqtgraph",
        "netCDF4",
        "pandas",
        "python-pptx",
        "Pillow",
        "pyvista",
        "pyvistaqt",
        "vtk"
    ],
    python_requires='>=3.8',
    extras_require={
        'docs': ['docutils<0.18,>=0.14', 'ipykernel', 'ipython',
                 'jinja2>=3.0', 'nbsphinx',
                 'sphinx', 'sphinx-rtd-theme', 'sphinxcontrib-bibtex',
                 'sphinx-copybutton'],
        'testing': ['pycodestyle', 'pylint', 'pytest', 'pytest-cov'],
        'all': ['docutils<0.18,>=0.14', 'ipykernel', 'ipython', 'jinja2>=3.0',
                'nbsphinx', 'pycodestyle', 'pylint', 'pytest', 'pytest-cov',
                'sphinx', 'sphinx-copybutton',
                'sphinx-rtd-theme', 'sphinxcontrib-bibtex']},
    classifiers=['Natural Language :: English',
                 'Operating System :: Microsoft :: Windows :: Windows 10',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: POSIX :: Linux',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Framework :: Pytest',
                 'License :: Other/Proprietary License'
                 ]
)
