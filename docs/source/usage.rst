Usage
=====

.. _installation:

Installation
------------

To use SDynPy, first download it from the 
`CEE-GitLab page <https://cee-gitlab.sandia.gov/dprohe/structural-dynamics-python-libraries>`_.
Once downloaded, you can install it by using :code:`pip`.

.. code-block:: console
    
   $ cd <download_directory_containing_setup.py>
   $ pip install .
   

To install the package and the testing dependencies all at once via `pip` one can run

::

    python3 -m pip install .[testing] -i https://nexus.web.sandia.gov/repository/pypi-group/simple

This will install the package as well as the dependencies required for testing.  The test-suite can then be run via

::

    pytest -m "not unwritten" tests/ --cov=src/sdynpy
    
Conda Install
~~~~~~~~~~~~~

After cloning the repo, you can setup a computational environment on your Linux blade with all the required dependencies for SDynPy by running the setup shell script::

    source setup_SDynPy.sh

This will load the latest Anaconda3 module, and install the various dependencies from the conda/pip sources on the Nexus Sandia server.

.. note::
    If not on your blade, you can still use the shell script (on Linux), but you will want to comment out the line `module load apps/anaconda3-2022.05`.  If you are working on a Windows OS, then you will want to open up a Conda terminal and simply run the conda commands referenced in the shell script.

.. note::

    If you utilized the `setup_SDynPy.sh` script, then the testing dependencies should also have been installed, so you can just execute the test commands as described above.
   
SDynPy Overview
---------------

With SDynPy installed, you can then import it into your python code to perform
analyes.

.. code-block:: python

   import sdynpy as sdpy

SDynPy Core Data Objects
~~~~~~~~~~~~~~~~~~~~~~~~

SDynPy provides five core data objects:

* :py:class:`sdpy.CoordinateArray<sdynpy.core.sdynpy_coordinate.CoordinateArray>` - An array of degrees of freedom (node and direction) e.g. '101X+' or '23RZ-'
* :py:class:`sdpy.NDDataArray<sdynpy.core.sdynpy_data.NDDataArray>` - An array of data functions.  Subclasses of this class provide functionality for specific types of data, e.g. :py:class:`time histories<sdynpy.core.sdynpy_data.TimeHistoryArray>`, :py:class:`frequency response functions<sdynpy.core.sdynpy_data.TransferFunctionArray>`, etc.
* :py:class:`sdpy.ShapeArray<sdynpy.core.sdynpy_shape.ShapeArray>` - An array of shapes which define deformations at degrees of freedom.
* :py:class:`sdpy.Geometry<sdynpy.core.sdynpy_geometry.Geometry>` - A class containing :py:class:`nodes<sdynpy.core.sdynpy_geometry.NodeArray>`, :py:class:`elements<sdynpy.core.sdynpy_geometry.ElementArray>`, :py:class:`lines<sdynpy.core.sdynpy_geometry.TracelineArray>`, and :py:class:`coordinate systems<sdynpy.core.sdynpy_geometry.CoordinateSystemArray>` that define the geometry of a test.
* :py:class:`sdpy.System<sdynpy.core.sdynpy_system.System>` - A class containing mass, stiffness, and damping matrices, as well as a transformation from the inner state to physical degrees of freedom (e.g. a mode shape matrix to transform from modal coordinates defined by a modal mass, stiffness, and damping matrix to physical coordinates), which is useful for time integration and substructuring.

SDynPy objects are generally built using subclasses of Numpy's `ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
class, which means SDynPy objects can generally use all the nice features of
that object, including broadcasting, as well as many of the Numpy functions
operating on ndarrays such as `intersect1d <https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html>`_.

Loading Test Data
~~~~~~~~~~~~~~~~~

SDynPy can read and write data to the `Universal File Format <https://www.ceas3.uc.edu/sdrluff/>`_
using the :py:mod:`sdpy.unv<sdynpy.fileio.sdynpy_uff>` module.  SDynPy can
currently read the following :py:mod:`datasets<sdynpy.fileio.sdynpy_uff_datasets>`:

* :py:mod:`55<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_55>` - Data at Nodes (:py:class:`sdpy.ShapeArray<sdynpy.core.sdynpy_shape.ShapeArray>`)
* :py:mod:`58<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_58>` - Function at Nodal DOF (:py:class:`sdpy.NDDataArray<sdynpy.core.sdynpy_data.NDDataArray>`)
* :py:mod:`82<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_82>` - Tracelines (:py:class:`sdpy.TracelineArray<sdynpy.core.sdynpy_geometry.TracelineArray>`)
* :py:mod:`151<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_151>` - Header (not currently used in any SDynPy objects)
* :py:mod:`164<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_164>` - Units (not currently used in any SDynPy objects)
* :py:mod:`1858<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_1858>` - Qualifiers for Dataset 58 (not currently used in any SDynPy objects)
* :py:mod:`2400<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2400>` - Model Header (not currently used in any SDynPy objects)
* :py:mod:`2411<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2411>` - Nodes (:py:class:`sdpy.NodeArray<sdynpy.core.sdynpy_geometry.NodeArray>`)
* :py:mod:`2412<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2412>` - Elements (:py:class:`sdpy.ElementArray<sdynpy.core.sdynpy_geometry.ElementArray>`)
* :py:mod:`2420<sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2420>` - Coordinate Systems (:py:class:`sdpy.CoordinateSystemArray<sdynpy.core.sdynpy_geometry.CoordinateSystemArray>`)

SDynPy will generally read in data using the :py:func:`sdpy.unv.readunv<sdynpy.fileio.sdynpy_uff.readunv>` function.
This will output data into a dictionary where the key is the dataset number and
the value is the information inside the dataset.  Many SDynPy objects have a
:code:`from_uff` function that when passed the universal file format dictionary
will automatically construct a SDynPy object from the data within.

.. code-block:: python
   
   # Read in the data from the UFF file
   uff_dict = sdpy.uff.readuff('path/to/uff/file.uff')
   # Parse the data in the dictionary into a SDynPy Geometry
   geometry = sdpy.Geometry.from_uff(uff_dict)
   
SDynPy can also read time data from `Rattlesnake <https://github.com/sandialabs/rattlesnake-vibration-controller>`_'s
netCDF output using the :py:func:`sdpy.rattlesnake.read_rattlesnake_output<sdynpy.fileio.sdynpy_rattlesnake.read_rattlesnake_output>`
function.  This function will return a :py:class:`sdpy.TimeHistoryArray<sdynpy.core.sdynpy_data.TimeHistoryArray>` object,
as well as a pandas `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_ object representing
the channel table.

Finally, SDynPy can also read data from Correlated Solutions' `VIC3D <https://www.correlatedsolutions.com/vic-3d/>`_ Digital Image Correlation software.  It assumes the data has
been exported to .mat files from the VIC3D software.  The :py:func:`sdpy.vic.read_vic3D_mat_files<sdynpy.fileio.sdynpy_vic.read_vic3D_mat_files>`
function can be given the list of .mat files from VIC3D, and it will automatically generate a test geometry and time history from those results.

.. code-block:: python

   from glob import glob
   
   # Get files
   files = glob(r'*.mat')
   
   # Read in time and displacement data
   geometry,time_data = sdpy.vic.read_vic3D_mat_files(files)

Finite Element Models
~~~~~~~~~~~~~~~~~~~~~

SDynPy also has capabilities to work with finite element models and data.

SDynPy can read or write to the `Exodus <https://www.osti.gov/servlets/purl/10102115>`_
file format.  It has two representations for Exodus models; the first is 
:py:class:`sdpy.Exodus<sdynpy.fem.sdynpy_exodus.Exodus>`, which keeps the file on disk and only reads and writes
what is requested to it.  This is most suitable for large files.  A second
way to interact with Exodus models is through the :py:class:`sdpy.ExodusInMemory<sdynpy.fem.sdynpy_exodus.ExodusInMemory>`
class, which, as the name suggests, reads the entire model into memory.

Similarly to the universal file format readers, the output :py:class:`sdpy.Exodus<sdynpy.fem.sdynpy_exodus.Exodus>` or 
:py:class:`sdpy.ExodusInMemory<sdynpy.fem.sdynpy_exodus.ExodusInMemory>` objects
can be transformed into SDynPy objects through various :code:`from_exodus` functions
in the SDynPy objects.

.. code-block:: python
   
   # Read in the data from the UFF file
   exo = sdpy.Exodus('path/to/exodus/file.exo')
   # Parse the data in the dictionary into a SDynPy Geometry
   geometry = sdpy.Geometry.from_exodus(exo)
   
SDynPy can create small beam finite element models using the :py:mod:`sdpy.beam<sdynpy.fem.sdynpy_beam>`
module.  Using the function :py:func:`sdpy.beam.beamkm<sdynpy.fem.sdynpy_beam.beamkm>` or its helper function for
2D beams :py:func:`sdpy.beam.beamkm_2d<sdynpy.fem.sdynpy_beam.beamkm_2d>` will produce
beam mass and stiffness matrices which can be used for finite element analysis.

SDynPy also has functionality for producing system matrices for
electro-mechanical modeling of shakers in :py:mod:`sdpy.shaker<sdynpy.fem.sdynpy_shaker>`.
These are expected to be substructured to another finite element model in order
to predict the voltage, current, and force required for a given test.

Degree of freedom optimization routines can be found in :py:mod:`sdpy.dof<sdynpy.fem.sdynpy_dof>`.
These include Effective Independence and Condition Number optimization routines.

Experimental Modal Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SDynPy has the ability to fit modes to structures using the Synthesize Modes and
Correlate :py:class:`sdpy.SMAC<sdynpy.modal.sdynpy_smac.SMAC>` or PolyMax
:py:class:`sdpy.PolyMax<sdynpy.modal.sdynpy_polymax.PolyMax>` routines.

Both SMAC and PolyMax have graphical user interfaces available to make the curve
fitting process easier (:py:class:`sdpy.SMAC_GUI<sdynpy.modal.sdynpy_smac.SMAC_GUI>`,
:py:class:`sdpy.PolyMax_GUI<sdynpy.modal.sdynpy_polymax.PolyMax_GUI>`).  These can be
run from an IPython console.

Included in SDynPy is interactive plotting capabilities where mode shapes can
be animated or several data sets can be plotted.

Documentation
~~~~~~~~~~~~~

SDynPy has the ability to automatically document portions of analysis by
exporting to a Microsoft PowerPoint presentation or LaTeX source code.  A
PowerPoint presentation can be created using the 
:py:func:`sdpy.doc.create_summary_pptx<sdynpy.doc.sdynpy_ppt.create_summary_pptx>`
function, or a LaTeX file can be created using the 
:py:func:`sdpy.doc.create_latex_summary<sdynpy.doc.sdynpy_latex.create_latex_summary>`
function.

Signal Processing
~~~~~~~~~~~~~~~~~

SDynPy has several general purpose signal processing tools as well.  These include:

* :py:mod:`sdpy.frf<sdynpy.signal_processing.sdynpy_frf>` - Functions for computing and working with Frequency Response Functions
* :py:mod:`sdpy.cpsd<sdynpy.signal_processing.sdynpy_cpsd>` - Functions for computing and working with Cross-Power Spectral Density matrices
* :py:mod:`sdpy.integration<sdynpy.signal_processing.sdynpy_integration>` - Functions for performing integration of equations of motion
* :py:mod:`sdpy.correlation<sdynpy.signal_processing.sdynpy_correlation>` - Functions for comparing data
* :py:mod:`sdpy.complex<sdynpy.signal_processing.sdynpy_complex>` - Functions for working with complex numbers
* :py:mod:`sdpy.rotation<sdynpy.signal_processing.sdynpy_rotation>` - Functions for computing and working with rotation matrices
* :py:mod:`sdpy.generator<sdynpy.signal_processing.sdynpy_generator>` - Functions for generating standard signal types such as sine or pseudorandom