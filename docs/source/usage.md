# Usage

## Installation

SDynPy can be installed with ``pip``

```{code-block} bash
pip install sdynpy
```

## SDynPy Overview


With SDynPy installed, you can then import it into your python code to perform
analyes.

```{code-block} python
import sdynpy as sdpy
```

### SDynPy Core Data Objects


SDynPy provides five core data objects:

* [`sdpy.CoordinateArray`](xref:api#sdynpy.core.sdynpy_coordinate.CoordinateArray) - An array of degrees of freedom (node and direction) e.g. '101X+' or '23RZ-'
* [`sdpy.NDDataArray`](xref:api#sdynpy.core.sdynpy_data.NDDataArray) - An array of data functions.  Subclasses of this class provide functionality for specific types of data, e.g. [`time histories`](xref:api#sdynpy.core.sdynpy_data.TimeHistoryArray), [`frequency response functions`](xref:api#sdynpy.core.sdynpy_data.TransferFunctionArray), etc.
* [`sdpy.ShapeArray`](xref:api#sdynpy.core.sdynpy_shape.ShapeArray) - An array of shapes which define deformations at degrees of freedom.
* [`sdpy.Geometry`](xref:api#sdynpy.core.sdynpy_geometry.Geometry) - A class containing [`nodes`](xref:api#sdynpy.core.sdynpy_geometry.NodeArray), [`elements`](xref:api#sdynpy.core.sdynpy_geometry.ElementArray), [`lines`](xref:api#sdynpy.core.sdynpy_geometry.TracelineArray), and [`coordinate systems`](xref:api#sdynpy.core.sdynpy_geometry.CoordinateSystemArray) that define the geometry of a test.
* [`sdpy.System`](xref:api#sdynpy.core.sdynpy_system.System) - A class containing mass, stiffness, and damping matrices, as well as a transformation from the inner state to physical degrees of freedom (e.g. a mode shape matrix to transform from modal coordinates defined by a modal mass, stiffness, and damping matrix to physical coordinates), which is useful for time integration and substructuring.

SDynPy objects are generally built using subclasses of Numpy's [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)
class, which means SDynPy objects can generally use all the nice features of
that object, including broadcasting, as well as many of the Numpy functions
operating on ndarrays such as [intersect1d](https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html).

### Loading Test Data


SDynPy can read and write data to the [Universal File Format](https://www.ceas3.uc.edu/sdrluff/)
using the [`sdpy.unv`](xref:api#sdynpy.fileio.sdynpy_uff) module.  SDynPy can
currently read the following [`datasets`](xref:api#sdynpy.fileio.sdynpy_uff_datasets):

* [`55`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_55) - Data at Nodes ([`sdpy.ShapeArray`](xref:api#sdynpy.core.sdynpy_shape.ShapeArray))
* [`58`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_58) - Function at Nodal DOF ([`sdpy.NDDataArray`](xref:api#sdynpy.core.sdynpy_data.NDDataArray))
* [`82`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_82) - Tracelines ([`sdpy.TracelineArray`](xref:api#sdynpy.core.sdynpy_geometry.TracelineArray))
* [`151`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_151) - Header (not currently used in any SDynPy objects)
* [`164`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_164) - Units (not currently used in any SDynPy objects)
* [`1858`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_1858) - Qualifiers for Dataset 58 (not currently used in any SDynPy objects)
* [`2400`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2400) - Model Header (not currently used in any SDynPy objects)
* [`2411`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2411) - Nodes ([`sdpy.NodeArray`](xref:api#sdynpy.core.sdynpy_geometry.NodeArray))
* [`2412`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2412) - Elements ([`sdpy.ElementArray`](xref:api#sdynpy.core.sdynpy_geometry.ElementArray))
* [`2420`](xref:api#sdynpy.fileio.sdynpy_uff_datasets.sdynpy_uff_dataset_2420) - Coordinate Systems ([`sdpy.CoordinateSystemArray`](xref:api#sdynpy.core.sdynpy_geometry.CoordinateSystemArray))

SDynPy will generally read in data using the [`sdpy.unv.readunv`](xref:api#sdynpy.fileio.sdynpy_uff.readunv) function.
This will output data into a dictionary where the key is the dataset number and
the value is the information inside the dataset.  Many SDynPy objects have a
:code:`from_uff` function that when passed the universal file format dictionary
will automatically construct a SDynPy object from the data within.

:::{code-block} python
# Read in the data from the UFF file
uff_dict = sdpy.uff.readuff('path/to/uff/file.uff')
# Parse the data in the dictionary into a SDynPy Geometry
geometry = sdpy.Geometry.from_uff(uff_dict)
:::
   
SDynPy can also read time data from [Rattlesnake](https://github.com/sandialabs/rattlesnake-vibration-controller)'s
netCDF output using the [`sdpy.rattlesnake.read_rattlesnake_output`](xref:api#sdynpy.fileio.sdynpy_rattlesnake.read_rattlesnake_output)
function.  This function will return a [`sdpy.TimeHistoryArray`](xref:api#sdynpy.core.sdynpy_data.TimeHistoryArray) object,
as well as a pandas [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) object representing
the channel table.

Finally, SDynPy can also read data from Correlated Solutions' [VIC3D](https://www.correlatedsolutions.com/vic-3d/) Digital Image Correlation software.  It assumes the data has
been exported to .mat files from the VIC3D software.  The [`sdpy.vic.read_vic3D_mat_files`](xref:api#sdynpy.fileio.sdynpy_vic.read_vic3D_mat_files)
function can be given the list of .mat files from VIC3D, and it will automatically generate a test geometry and time history from those results.

:::{code-block} python
from glob import glob

# Get files
files = glob(r'*.mat')

# Read in time and displacement data
geometry,time_data = sdpy.vic.read_vic3D_mat_files(files)
:::

### Finite Element Models

SDynPy also has capabilities to work with finite element models and data.

SDynPy can read or write to the [Exodus](https://www.osti.gov/servlets/purl/10102115)
file format.  It has two representations for Exodus models; the first is 
[`sdpy.Exodus`](xref:api#sdynpy.fem.sdynpy_exodus.Exodus), which keeps the file on disk and only reads and writes
what is requested to it.  This is most suitable for large files.  A second
way to interact with Exodus models is through the [`sdpy.ExodusInMemory`](xref:api#sdynpy.fem.sdynpy_exodus.ExodusInMemory)
class, which, as the name suggests, reads the entire model into memory.

Similarly to the universal file format readers, the output [`sdpy.Exodus`](xref:api#sdynpy.fem.sdynpy_exodus.Exodus) or 
[`sdpy.ExodusInMemory`](xref:api#sdynpy.fem.sdynpy_exodus.ExodusInMemory) objects
can be transformed into SDynPy objects through various :code:`from_exodus` functions
in the SDynPy objects.

:::{code-block} python
# Read in the data from the UFF file
exo = sdpy.Exodus('path/to/exodus/file.exo')
# Parse the data in the dictionary into a SDynPy Geometry
geometry = sdpy.Geometry.from_exodus(exo)
:::
   
SDynPy can create small beam finite element models using the [`sdpy.beam`](xref:api#sdynpy.fem.sdynpy_beam)
module.  Using the function [`sdpy.beam.beamkm`](xref:api#sdynpy.fem.sdynpy_beam.beamkm) or its helper function for
2D beams [`sdpy.beam.beamkm_2d`](xref:api#sdynpy.fem.sdynpy_beam.beamkm_2d) will produce
beam mass and stiffness matrices which can be used for finite element analysis.

SDynPy also has functionality for producing system matrices for
electro-mechanical modeling of shakers in [`sdpy.shaker`](xref:api#sdynpy.fem.sdynpy_shaker).
These are expected to be substructured to another finite element model in order
to predict the voltage, current, and force required for a given test.

Degree of freedom optimization routines can be found in [`sdpy.dof`](xref:api#sdynpy.fem.sdynpy_dof).
These include Effective Independence and Condition Number optimization routines.

### Experimental Modal Analysis

SDynPy has the ability to fit modes to structures using the Synthesize Modes and
Correlate [`sdpy.SMAC`](xref:api#sdynpy.modal.sdynpy_smac.SMAC) or PolyPy
[`sdpy.PolyPy`](xref:api#sdynpy.modal.sdynpy_polypy.PolyPy) routines.

Both SMAC and PolyPy have graphical user interfaces available to make the curve
fitting process easier ([`sdpy.SMAC_GUI`](xref:api#sdynpy.modal.sdynpy_smac.SMAC_GUI),
[`sdpy.PolyPy_GUI`](xref:api#sdynpy.modal.sdynpy_polypy.PolyPy_GUI)).  These can be
run from an IPython console.

Included in SDynPy is interactive plotting capabilities where mode shapes can
be animated or several data sets can be plotted.

### Documentation

SDynPy has the ability to automatically document portions of analysis by
exporting to a Microsoft PowerPoint presentation or LaTeX source code.  A
PowerPoint presentation can be created using the 
[`sdpy.doc.create_summary_pptx`](xref:api#sdynpy.doc.sdynpy_ppt.create_summary_pptx)
function, or a LaTeX file can be created using the 
[`sdpy.doc.create_latex_summary`](xref:api#sdynpy.doc.sdynpy_latex.create_latex_summary)
function.

### Signal Processing

SDynPy has several general purpose signal processing tools as well.  These include:

* [`sdpy.frf`](xref:api#sdynpy.signal_processing.sdynpy_frf) - Functions for computing and working with Frequency Response Functions
* [`sdpy.cpsd`](xref:api#sdynpy.signal_processing.sdynpy_cpsd) - Functions for computing and working with Cross-Power Spectral Density matrices
* [`sdpy.integration`](xref:api#sdynpy.signal_processing.sdynpy_integration) - Functions for performing integration of equations of motion
* [`sdpy.correlation`](xref:api#sdynpy.signal_processing.sdynpy_correlation) - Functions for comparing data
* [`sdpy.complex`](xref:api#sdynpy.signal_processing.sdynpy_complex) - Functions for working with complex numbers
* [`sdpy.rotation`](xref:api#sdynpy.signal_processing.sdynpy_rotation) - Functions for computing and working with rotation matrices
* [`sdpy.generator`](xref:api#sdynpy.signal_processing.sdynpy_generator) - Functions for generating standard signal types such as sine or pseudorandom
* [`sdpy.camera`](xref:api#sdynpy.signal_processing.sdynpy_camera) - Functions for working with camera projections and projections
* [`sdpy.harmonic`](xref:api#sdynpy.signal_processing.sdynpy_harmonic) - Functions for working with harmonic signals
* [`sdpy.geometry_fitting`](xref:api#sdynpy.signal_processing.sdynpy_geometry_fitting) - Functions for fitting geometry to data
* [`sdpy.frf_inverse`](xref:api#sdynpy.signal_processing.sdynpy_frf_inverse) - Functions for inverting frequency response functions for inverse problems
* [`sdpy.srs`](xref:api#sdynpy.signal_processing.sdynpy_srs) - Functions for computing shock response spectra and generating sum-decayed-sines signals