# Package Layout

This document will describe the layout of SDynPy to help users find the functions they are looking for.

## SDynPy Subpackages

SDynPy is split up into several subpackages to help organize its content.  While these subpackages are convenient to organize the content, they are not convenient to use, as typical function calls might require descending multiple package levels, resulting in inconveniently long code.  Therefore, SDynPy heavily relies on aliases to commonly used functions and classes.

The following sections describe the major subpackages that exist in SDynPy.

## SDynPy Core Objects

The [`sdynpy.core`](xref:api#sdynpy.core) subpackage contains Python modules that define the core SDynPy objects and other core functionality.  These are designed to represent common data used in structural dynamics, including degrees of freedom, geometry, data, or dynamic systems.  Because of their ubiquitous nature, almost all of the modules, classes, and their corresponding helper functions are aliased to the top-level SDynPy namespace.  The [`core`](xref:api#sdynpy.core) modules are:

| SDynPy Core Modules                             | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_array`](xref:api#sdynpy.core.sdynpy_array)        | [`sdynpy.array`](xref:api#sdynpy.core.sdynpy_array)       | Definition of the parent [`SdynpyArray`](xref:api#sdynpy.core.sdynpy_array.SdynpyArray) class.    |
| [`sdynpy.core.sdynpy_colors`](xref:api#sdynpy.core.sdynpy_colors)       | [`sdynpy.colors`](xref:api#sdynpy.core.sdynpy_colors)      | Definition of the color map scheme used by SDynPy.    
| [`sdynpy.core.sdynpy_coordinate`](xref:api#sdynpy.core.sdynpy_coordinate)   | [`sdynpy.coordinate`](xref:api#sdynpy.core.sdynpy_coordinate)  | Definition of the [`CoordinateArray`](xref:api#sdynpy.core.sdynpy_coordinate.CoordinateArray) class used to represent degrees of freedom.    |
| [`sdynpy.core.sdynpy_data`](xref:api#sdynpy.core.sdynpy_data)         | [`sdynpy.data`](xref:api#sdynpy.core.sdynpy_data)        | Definition of the [`NDDataArray`](xref:api#sdynpy.core.sdynpy_data.NDDataArray) class and it's subclasses. ([`TimeHistoryArray`](xref:api#sdynpy.core.sdynpy_data.TimeHistoryArray), [`SpectrumArray`](xref:api#sdynpy.core.sdynpy_data.SpectrumArray), etc.) used to represent data. |
| [`sdynpy.core.sdynpy_geometry`](xref:api#sdynpy.core.sdynpy_geometry)     | [`sdynpy.geometry`](xref:api#sdynpy.core.sdynpy_geometry)    | Definition of the [`Geometry`](xref:api#sdynpy.core.sdynpy_geometry.Geometry) class and its component classes ([`NodeArray`](xref:api#sdynpy.core.sdynpy_geometry.NodeArray), [`ElementArray`](xref:api#sdynpy.core.sdynpy_geometry.ElementArray), [`TracelineArray`](xref:api#sdynpy.core.sdynpy_geometry.TracelineArray), [`CoordinateSystemArray`](xref:api#sdynpy.core.sdynpy_geometry.CoordinateSystemArray)) used to represent the locations and orientations of measurements.   |
| [`sdynpy.core.sdynpy_matrix`](xref:api#sdynpy.core.sdynpy_matrix)       | [`sdynpy.matrix_mod`](xref:api#sdynpy.core.sdynpy_matrix)  | Definition of the [`Matrix`](xref:api#sdynpy.core.sdynpy_matrix.Matrix) class used to map degrees of freedom to rows and columns of a matrix.    |
| [`sdynpy.core.sdynpy_shape`](xref:api#sdynpy.core.sdynpy_shape)        | [`sdynpy.shape`](xref:api#sdynpy.core.sdynpy_shape)       | Definition of the parent [`ShapeArray`](xref:api#sdynpy.core.sdynpy_shape.ShapeArray) class used to represent mode or deflection shapes.    |
| [`sdynpy.core.sdynpy_system`](xref:api#sdynpy.core.sdynpy_system)       | [`sdynpy.system`](xref:api#sdynpy.core.sdynpy_system)      | Definition of the parent [`System`](xref:api#sdynpy.core.sdynpy_system.System) class used to represent dynamic systems (mass, stiffness, and damping).    |

In addition to the modules being aliased to the SDynPy namespace, many of the classes and functions from the [`core`](xref:api#sdynpy.core) subpackage are also aliased to the top-level namespace, as these are the most often called functions and classes in SDynPy.


| SDynPy Array Class               | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_array.SdynpyArray`](xref:api#sdynpy.core.sdynpy_array.SdynpyArray)        | [`sdynpy.SdynpyArray`](xref:api#sdynpy.core.sdynpy_array.SdynpyArray)       | Parent class of all SDynPy Arrays. |

| Coordinate Classes and Functions  | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_coordinate.CoordinateArray`](xref:api#sdynpy.core.sdynpy_coordinate.CoordinateArray)   | [`sdynpy.CoordinateArray`](xref:api#sdynpy.core.sdynpy_coordinate.CoordinateArray)  | Class to represent degrees of freedom.    |
| [`sdynpy.core.sdynpy_coordinate.coordinate_array`](xref:api#sdynpy.core.sdynpy_coordinate.coordinate_array)   | [`sdynpy.coordinate_array`](xref:api#sdynpy.core.sdynpy_coordinate.coordinate_array)  | Helper function to create [`CoordinateArray`](xref:api#sdynpy.core.sdynpy_coordinate.CoordinateArray) objects.    |

     
| Data Classes and Functions              | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_data.NDDataArray`](xref:api#sdynpy.core.sdynpy_data.NDDataArray)         | [`sdynpy.NDDataArray`](xref:api#sdynpy.core.sdynpy_data.NDDataArray)        | Parent class of all SDynPy Data Arrays.  |
| [`sdynpy.core.sdynpy_data.TimeHistoryArray`](xref:api#sdynpy.core.sdynpy_data.TimeHistoryArray)         | [`sdynpy.TimeHistoryArray`](xref:api#sdynpy.core.sdynpy_data.TimeHistoryArray)        | Data class representing time histories.  |
| [`sdynpy.core.sdynpy_data.TransferFunctionArray`](xref:api#sdynpy.core.sdynpy_data.TransferFunctionArray)         | [`sdynpy.TransferFunctionArray`](xref:api#sdynpy.core.sdynpy_data.TransferFunctionArray)        | Data class representing transfer functions or frequency response functions.  |
| [`sdynpy.core.sdynpy_data.CoherenceArray`](xref:api#sdynpy.core.sdynpy_data.CoherenceArray)         | [`sdynpy.CoherenceArray`](xref:api#sdynpy.core.sdynpy_data.CoherenceArray)        | Data class representing coherence functions.  |
| [`sdynpy.core.sdynpy_data.MultipleCoherenceArray`](xref:api#sdynpy.core.sdynpy_data.MultipleCoherenceArray)         | [`sdynpy.MultipleCoherenceArray`](xref:api#sdynpy.core.sdynpy_data.MultipleCoherenceArray)        | Data class representing multiple coherence functions.  |
| [`sdynpy.core.sdynpy_data.PowerSpectralDensityArray`](xref:api#sdynpy.core.sdynpy_data.PowerSpectralDensityArray)         | [`sdynpy.PowerSpectralDensityArray`](xref:api#sdynpy.core.sdynpy_data.PowerSpectralDensityArray)        | Data class representing power spectral density functions.  |
| [`sdynpy.core.sdynpy_data.SpectrumArray`](xref:api#sdynpy.core.sdynpy_data.SpectrumArray)         | [`sdynpy.SpectrumArray`](xref:api#sdynpy.core.sdynpy_data.SpectrumArray)        | Data class representing spectra (e.g. FFTs).  |
| [`sdynpy.core.sdynpy_data.GUIPlot`](xref:api#sdynpy.core.sdynpy_data.GUIPlot)         | [`sdynpy.GUIPlot`](xref:api#sdynpy.core.sdynpy_data.GUIPlot)        | An interactive data plotter.  |
| [`sdynpy.core.sdynpy_data.CPSDPlot`](xref:api#sdynpy.core.sdynpy_data.CPSDPlot)         | [`sdynpy.CPSDPlot`](xref:api#sdynpy.core.sdynpy_data.CPSDPlot)        | An interactive data plotter specifically for cross-power spectral density data.  |
| [`sdynpy.core.sdynpy_data.data_array`](xref:api#sdynpy.core.sdynpy_data.data_array)         | [`sdynpy.data_array`](xref:api#sdynpy.core.sdynpy_data.data_array)        | Helper function to create [`NDDataArray`](xref:api#sdynpy.core.sdynpy_data.NDDataArray) (and subclasses) objects.  |
| [`sdynpy.core.sdynpy_data.time_history_array`](xref:api#sdynpy.core.sdynpy_data.time_history_array)         | [`sdynpy.time_history_array`](xref:api#sdynpy.core.sdynpy_data.time_history_array)        | Helper function to create [`TimeHistoryArray`](xref:api#sdynpy.core.sdynpy_data.TimeHistoryArray) objects.  |
| [`sdynpy.core.sdynpy_data.transfer_function_array`](xref:api#sdynpy.core.sdynpy_data.transfer_function_array)         | [`sdynpy.transfer_function_array`](xref:api#sdynpy.core.sdynpy_data.transfer_function_array)        | Helper function to create [`TransferFunctionArray`](xref:api#sdynpy.core.sdynpy_data.TransferFunctionArray) objects.  |
| [`sdynpy.core.sdynpy_data.coherence_array`](xref:api#sdynpy.core.sdynpy_data.coherence_array)         | [`sdynpy.coherence_array`](xref:api#sdynpy.core.sdynpy_data.coherence_array)        | Helper function to create [`CoherenceArray`](xref:api#sdynpy.core.sdynpy_data.CoherenceArray) objects.  |
| [`sdynpy.core.sdynpy_data.multiple_coherence_array`](xref:api#sdynpy.core.sdynpy_data.multiple_coherence_array)         | [`sdynpy.multiple_coherence_array`](xref:api#sdynpy.core.sdynpy_data.multiple_coherence_array)        | Helper function to create [`MultipleCoherenceArray`](xref:api#sdynpy.core.sdynpy_data.MultipleCoherenceArray) objects.  |
| [`sdynpy.core.sdynpy_data.power_spectral_density_array`](xref:api#sdynpy.core.sdynpy_data.power_spectral_density_array)         | [`sdynpy.power_spectral_density_array`](xref:api#sdynpy.core.sdynpy_data.power_spectral_density_array)        | Helper function to create [`PowerSpectralDensityArray`](xref:api#sdynpy.core.sdynpy_data.PowerSpectralDensityArray) objects.  |
| [`sdynpy.core.sdynpy_data.spectrum_array`](xref:api#sdynpy.core.sdynpy_data.spectrum_array)         | [`sdynpy.spectrum_array`](xref:api#sdynpy.core.sdynpy_data.spectrum_array)        | Helper function to create [`SpectrumArray`](xref:api#sdynpy.core.sdynpy_data.SpectrumArray) objects.  |
     
     
| Geometry Classes and Functions  | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_geometry.Geometry`](xref:api#sdynpy.core.sdynpy_geometry.Geometry)     | [`sdynpy.Geometry`](xref:api#sdynpy.core.sdynpy_geometry.Geometry)    | Class to represent a test or analysis geometry.   |
| [`sdynpy.core.sdynpy_geometry.NodeArray`](xref:api#sdynpy.core.sdynpy_geometry.NodeArray)     | [`sdynpy.NodeArray`](xref:api#sdynpy.core.sdynpy_geometry.NodeArray)    | Class to represent node locations in a geometry.   |
| [`sdynpy.core.sdynpy_geometry.CoordinateSystemArray`](xref:api#sdynpy.core.sdynpy_geometry.CoordinateSystemArray)     | [`sdynpy.CoordinateSystemArray`](xref:api#sdynpy.core.sdynpy_geometry.CoordinateSystemArray)    | Class to represent global and local coordinate systems in a geometry.   |
| [`sdynpy.core.sdynpy_geometry.TracelineArray`](xref:api#sdynpy.core.sdynpy_geometry.TracelineArray)     | [`sdynpy.TracelineArray`](xref:api#sdynpy.core.sdynpy_geometry.TracelineArray)    | Class to represent lines connecting nodes for visualization.   |
| [`sdynpy.core.sdynpy_geometry.ElementArray`](xref:api#sdynpy.core.sdynpy_geometry.ElementArray)     | [`sdynpy.ElementArray`](xref:api#sdynpy.core.sdynpy_geometry.ElementArray)    | Class to represent elements connecting nodes for visualization.   |
| [`sdynpy.core.sdynpy_geometry.node_array`](xref:api#sdynpy.core.sdynpy_geometry.node_array)         | [`sdynpy.node_array`](xref:api#sdynpy.core.sdynpy_geometry.node_array)        | Helper function to create [`NodeArray`](xref:api#sdynpy.core.sdynpy_geometry.NodeArray) objects.  |
| [`sdynpy.core.sdynpy_geometry.coordinate_system_array`](xref:api#sdynpy.core.sdynpy_geometry.coordinate_system_array)         | [`sdynpy.coordinate_system_array`](xref:api#sdynpy.core.sdynpy_geometry.coordinate_system_array)        | Helper function to create [`CoordinateSystemArray`](xref:api#sdynpy.core.sdynpy_geometry.CoordinateSystemArray) objects.  |
| [`sdynpy.core.sdynpy_geometry.traceline_array`](xref:api#sdynpy.core.sdynpy_geometry.traceline_array)         | [`sdynpy.traceline_array`](xref:api#sdynpy.core.sdynpy_geometry.traceline_array)        | Helper function to create [`TracelineArray`](xref:api#sdynpy.core.sdynpy_geometry.TracelineArray) objects.  |
| [`sdynpy.core.sdynpy_geometry.element_array`](xref:api#sdynpy.core.sdynpy_geometry.element_array)         | [`sdynpy.element_array`](xref:api#sdynpy.core.sdynpy_geometry.element_array)        | Helper function to create [`ElementArray`](xref:api#sdynpy.core.sdynpy_geometry.ElementArray) objects.  |
| [`sdynpy.core.sdynpy_geometry.id_map`](xref:api#sdynpy.core.sdynpy_geometry.id_map)         | [`sdynpy.id_map`](xref:api#sdynpy.core.sdynpy_geometry.id_map)        | Class to represent identification number maps between two geometries.  |
  
| Matrix Classes and Functions     | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_matrix.Matrix`](xref:api#sdynpy.core.sdynpy_matrix.Matrix)       | [`sdynpy.Matrix`](xref:api#sdynpy.core.sdynpy_matrix.Matrix)  | Class used to map degrees of freedom to rows and columns of a matrix.    |
| [`sdynpy.core.sdynpy_matrix.matrix`](xref:api#sdynpy.core.sdynpy_matrix.matrix)       | [`sdynpy.matrix`](xref:api#sdynpy.core.sdynpy_matrix.matrix)  | Helper function to create [`Matrix`](xref:api#sdynpy.core.sdynpy_matrix.Matrix) objects.    |
     
     
| Shape Classes and Functions        | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_shape.ShapeArray`](xref:api#sdynpy.core.sdynpy_shape.ShapeArray)       | [`sdynpy.ShapeArray`](xref:api#sdynpy.core.sdynpy_shape.ShapeArray)  | Class used to represent mode or deflection shapes.    |
| [`sdynpy.core.sdynpy_shape.shape_array`](xref:api#sdynpy.core.sdynpy_shape.shape_array)       | [`sdynpy.shape_array`](xref:api#sdynpy.core.sdynpy_shape.shape_array)  | Helper function to create [`ShapeArray`](xref:api#sdynpy.core.sdynpy_shape.ShapeArray) objects.    |
     
     
| System Classes            | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.core.sdynpy_system.System`](xref:api#sdynpy.core.sdynpy_system.System)       | [`sdynpy.System`](xref:api#sdynpy.core.sdynpy_system.System)      | Class used to represent dynamic systems with mass, stiffness, and damping matrices.    |


## File Input and Output

Structural dynamics data often comes from external sources, whether it is a modal, vibration, or shock test or an equivalent finite element simulation.  Therefore, being able to quickly and easily bring external data into SDynPy is a priority.  While users could strip data from their external files and manually construct SDynPy objects from that data, this risks bookkeeping and other translation errors.  Therefore if a file type is commonly read into or written from SDynPy, it is useful to add a dedicated reader or writer into SDynPy to handle this translation correctly.  The [`sdynpy.fileio`](xref:api#sdynpy.fileio) subpackage contains much of the code to handle these conversions.  For convenience, the modules are aliased to the top-level namespace.  The [`fileio`](xref:api#sdynpy.fileio) modules are:

     
| File Input/Output Modules                         | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.fileio.sdynpy_uff`](xref:api#sdynpy.fileio.sdynpy_uff)        | [`sdynpy.uff`](xref:api#sdynpy.fileio.sdynpy_uff) or [`sdynpy.unv`](xref:api#sdynpy.fileio.sdynpy_uff)       | Functionality for reading and writing the [Universal File Format](https://www.ceas3.uc.edu/sdrluff/), a text-based file format common in structural dynamics.    |
| [`sdynpy.fileio.sdynpy_rattlesnake`](xref:api#sdynpy.fileio.sdynpy_rattlesnake)  | [`sdynpy.rattlesnake`](xref:api#sdynpy.fileio.sdynpy_rattlesnake)  | Functionality for reading output from the open-source vibration controller and modal testing software [Rattlesnake](https://github.com/sandialabs/rattlesnake-vibration-controller)  |
| [`sdynpy.fileio.sdynpy_vic`](xref:api#sdynpy.fileio.sdynpy_vic)  | [`sdynpy.vic`](xref:api#sdynpy.fileio.sdynpy_vic)  | Functionality for reading output from [Correlated Solution's VIC3D](https://www.correlatedsolutions.com/vic-3d) ``.mat`` file output.  |
| [`sdynpy.fileio.sdynpy_pdf3D`](xref:api#sdynpy.fileio.sdynpy_pdf3D)  | [`sdynpy.pdf3D`](xref:api#sdynpy.fileio.sdynpy_pdf3D)  | Functionality for writing geometry and shape data to a format that can be embedded into an [interactive PDF](https://helpx.adobe.com/acrobat/using/adding-3d-models-pdfs-acrobat.html) for test or analysis documentation.   |
| [`sdynpy.fileio.sdynpy_tshaker`](xref:api#sdynpy.fileio.sdynpy_tshaker)  | [`sdynpy.tshaker`](xref:api#sdynpy.fileio.sdynpy_tshaker)  | Functionality for reading output data from T-Shaker, a vibration shaker controller.  |
| [`sdynpy.fileio.sdynpy_escdf`](xref:api#sdynpy.fileio.sdynpy_escdf)  | [`sdynpy.escdf`](xref:api#sdynpy.fileio.sdynpy_escdf)  | Functionality for reading and writing the Engineering Sciences Common Data Format.  |

## Finite Elements and Similar Numerical Functionality

SDynPy has a limited set of finite element and other numerical functionality in the [`sdynpy.fem`](xref:api#sdynpy.fem) subpackage.  This includes simple beam finite elements, electrodynamic shaker models, and sensor optimization routines to select optimal sensors for a test from finite element results.  For convenience, the modules are aliased to the top-level namespace.  The [`fem`](xref:api#sdynpy.fem) modules are:

     
| FEM Modules                             | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.fem.sdynpy_beam`](xref:api#sdynpy.fem.sdynpy_beam)          | [`sdynpy.beam`](xref:api#sdynpy.fem.sdynpy_beam)       | Functionality for defining beam finite elements.  |
| [`sdynpy.fem.sdynpy_shaker`](xref:api#sdynpy.fem.sdynpy_shaker)        | [`sdynpy.shaker`](xref:api#sdynpy.fem.sdynpy_shaker)     | Functionality for definining shaker electromechanical models per [Lang and Snyder](http://www.sandv.com/downloads/0110lang.pdf). |
| [`sdynpy.fem.sdynpy_dof`](xref:api#sdynpy.fem.sdynpy_dof)           | [`sdynpy.dof`](xref:api#sdynpy.fem.sdynpy_dof)        | Techniques such as effective independence used to select sensors for a test given finite element data  |
| [`sdynpy.fem.sdynpy_exodus`](xref:api#sdynpy.fem.sdynpy_exodus)        | See Below           | Functionality for reading and writing the [Exodus](https://sandialabs.github.io/seacas-docs/exodusII-new.pdf) finite element model format.  |


Because the Exodus file format is used often at Sandia National Laboratories where SDynPy was originally developed, key classes from the [`sdynpy_exodus`](xref:api#sdynpy.fem.sdynpy_exodus) module are also aliased to the top-level namespace.

     
| Exodus Classes       | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.fem.sdynpy_exodus.Exodus`](xref:api#sdynpy.fem.sdynpy_exodus.Exodus)       | [`sdynpy.Exodus`](xref:api#sdynpy.fem.sdynpy_exodus.Exodus)  | Class to represent an Exodus file as stored on the filesystem    |
| [`sdynpy.fem.sdynpy_exodus.ExodusInMemory`](xref:api#sdynpy.fem.sdynpy_exodus.ExodusInMemory)       | [`sdynpy.ExodusInMemory`](xref:api#sdynpy.fem.sdynpy_exodus.ExodusInMemory)  | Class that represents an Exodus file in memory in a format similar to a Matlab Structure    |

## Modal Analysis

SDynPy has capabilities for performing experimental modal analysis which entails fitting modes to frequency response functions measured on the test article.  These and similar capabilities exist in the [`sdynpy.modal`](xref:api#sdynpy.modal) subpackage.  Many of the modules in this subpackage provide both code-based and GUI-based tools to fit modes to data.  In the case of the [`modal`](xref:api#sdynpy.modal) subpackage, much of the useful content boils down to a handful of classes.  Therefore, instead of aliasing the modules in the subpackage to the top-level namespace, SDynPy aliases these classes to the top-level namespace.

     
| Modal Classes | Alias | Description                                      |
|---|---|---|
| [`sdynpy.modal.sdynpy_polypy.PolyPy`](xref:api#sdynpy.modal.sdynpy_polypy.PolyPy)  | [`sdynpy.PolyPy`](xref:api#sdynpy.modal.sdynpy_polypy.PolyPy)     | Code-based implementation of the PolyMax mode fitter.  |
| [`sdynpy.modal.sdynpy_polypy.PolyPy_GUI`](xref:api#sdynpy.modal.sdynpy_polypy.PolyPy_GUI)  | [`sdynpy.PolyPy_GUI`](xref:api#sdynpy.modal.sdynpy_polypy.PolyPy_GUI)     | GUI-based implementation of the PolyMax mode fitter.  |
| [`sdynpy.modal.sdynpy_smac.SMAC`](xref:api#sdynpy.modal.sdynpy_smac.SMAC)  | [`sdynpy.SMAC`](xref:api#sdynpy.modal.sdynpy_smac.SMAC)  | Code-based implemenation of the Synthesize Modes and Correlate mode fitter.  |
| [`sdynpy.modal.sdynpy_smac.SMAC_GUI`](xref:api#sdynpy.modal.sdynpy_smac.SMAC_GUI)  | [`sdynpy.SMAC_GUI`](xref:api#sdynpy.modal.sdynpy_smac.SMAC_GUI)  | GUI-based implemenation of the Synthesize Modes and Correlate mode fitter.  |
| [`sdynpy.modal.sdynpy_ccmif.ColoredCMIF`](xref:api#sdynpy.modal.sdynpy_ccmif.ColoredCMIF)  | [`sdynpy.ColoredCMIF`](xref:api#sdynpy.modal.sdynpy_ccmif.ColoredCMIF)  | GUI-based tool to interactively select and combine modes fit to multiple single-reference measurements.  |
| [`sdynpy.modal.sdynpy_signal_processing_gui.SignalProcessingGUI`](xref:api#sdynpy.modal.sdynpy_signal_processing_gui.SignalProcessingGUI)  | [`sdynpy.SignalProcessingGUI`](xref:api#sdynpy.modal.sdynpy_signal_processing_gui.SignalProcessingGUI)  | GUI-based tool to compute spectral quantities from time histories.  |
| [`sdynpy.modal.sdynpy_modal_test.ModalTest`](xref:api#sdynpy.modal.sdynpy_modal_test.ModalTest)  | [`sdynpy.ModalTest`](xref:api#sdynpy.modal.sdynpy_modal_test.ModalTest)  | A class to represent a typical modal test, intended to aid modal test practitioners in the data processing and documentation workflow.  |

## Signal Processing

The [`sdynpy.signal_processing`](xref:api#sdynpy.signal_processing) subpackage includes a wide variety of tools that are related to structural dynamics.  Often, the code in this package is at a "lower level" than the rest of the SDynPy code, operating on raw data rather than SDynPy objects.  Many times, SDynPy objects will often call these lower-level functions as part of their own operations.  [`signal_processing`](xref:api#sdynpy.signal_processing) functions can be faster in that they remove the bookkeeping overhead that a lot of SDynPy objects implement, but the user must then be aware that their data must be correctly sorted prior to using the techniques.  Additionally, some of the content in [`signal_processing`](xref:api#sdynpy.signal_processing) does not fit well within the existing SDynPy core objects, so the subpackage serves as a kind of catch-all for tools that can be used in structural dynamics, but perhaps are not commonly used.  The modules of [`signal_processing`](xref:api#sdynpy.signal_processing) are generally aliased to the top-level namespace.

     
| Signal Processing Modules                             | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.signal_processing.sdynpy_camera`](xref:api#sdynpy.signal_processing.sdynpy_camera)  | [`sdynpy.camera`](xref:api#sdynpy.signal_processing.sdynpy_camera)     | Various functions for dealing with data from cameras, including camera calibration.  |
| [`sdynpy.signal_processing.sdynpy_complex`](xref:api#sdynpy.signal_processing.sdynpy_complex)  | [`sdynpy.complex`](xref:api#sdynpy.signal_processing.sdynpy_complex)     | Utility functions for working with complex numbers.  |
| [`sdynpy.signal_processing.sdynpy_correlation`](xref:api#sdynpy.signal_processing.sdynpy_correlation)  | [`sdynpy.correlation`](xref:api#sdynpy.signal_processing.sdynpy_correlation)     | Functions like the modal assurance cirterion for comparing data.  |
| [`sdynpy.signal_processing.sdynpy_cpsd`](xref:api#sdynpy.signal_processing.sdynpy_cpsd)  | [`sdynpy.cpsd`](xref:api#sdynpy.signal_processing.sdynpy_cpsd)     | Various functions for dealing with cross-power spectral density matrices.  |
| [`sdynpy.signal_processing.sdynpy_frf`](xref:api#sdynpy.signal_processing.sdynpy_frf)  | [`sdynpy.frf`](xref:api#sdynpy.signal_processing.sdynpy_frf)     | Various functions computing frequency response functions.  |
| [`sdynpy.signal_processing.sdynpy_frf_inverse`](xref:api#sdynpy.signal_processing.sdynpy_frf_inverse)  | [`sdynpy.frf_inverse`](xref:api#sdynpy.signal_processing.sdynpy_frf_inverse)     | Various functions for inverting frequency response functions, including regularization.  |
| [`sdynpy.signal_processing.sdynpy_generator`](xref:api#sdynpy.signal_processing.sdynpy_generator)  | [`sdynpy.generator`](xref:api#sdynpy.signal_processing.sdynpy_generator)     | Various functions for generating common time signals, like pseudorandom or sine sweeps.  |
| [`sdynpy.signal_processing.sdynpy_geometry_fitting`](xref:api#sdynpy.signal_processing.sdynpy_geometry_fitting)  | [`sdynpy.geometry_fitting`](xref:api#sdynpy.signal_processing.sdynpy_geometry_fitting)     | Various functions for dealing with geometry; for example, fitting a shape to a point cloud, or finding intersection points of lines.  |
| [`sdynpy.signal_processing.sdynpy_harmonic`](xref:api#sdynpy.signal_processing.sdynpy_harmonic)  | [`sdynpy.harmonic`](xref:api#sdynpy.signal_processing.sdynpy_harmonic)     | Functions for fitting sine waves to data.  |
| [`sdynpy.signal_processing.sdynpy_integration`](xref:api#sdynpy.signal_processing.sdynpy_integration)  | [`sdynpy.integration`](xref:api#sdynpy.signal_processing.sdynpy_integration)     | Various functions for generating and integrating state space systems.  |
| [`sdynpy.signal_processing.sdynpy_lrm`](xref:api#sdynpy.signal_processing.sdynpy_lrm)  | [`sdynpy.lrm`](xref:api#sdynpy.signal_processing.sdynpy_lrm)     | Advanced FRF computation technique using local rational modeling.  |
| [`sdynpy.signal_processing.sdynpy_rotation`](xref:api#sdynpy.signal_processing.sdynpy_rotation)  | [`sdynpy.rotation`](xref:api#sdynpy.signal_processing.sdynpy_rotation)     | Various functions for dealing with rotations; rotation matrices, Rodrigues parameters, rigid body transformations.  |
| [`sdynpy.signal_processing.sdynpy_srs`](xref:api#sdynpy.signal_processing.sdynpy_srs)  | [`sdynpy.srs`](xref:api#sdynpy.signal_processing.sdynpy_srs)     | Various functions for working with shock response spectra.  |

## Documentation

SDynPy has limited functionality to create documentation for test or analysis results.  This functionality is currently a work-in-progress, but capability exists to dump common formats into PowerPoint or LaTeX documents.

     
| Documentation Modules                                | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.doc.sdynpy_ppt`](xref:api#sdynpy.doc.sdynpy_ppt)  | [`sdynpy.doc.ppt`](xref:api#sdynpy.doc.sdynpy_ppt)     | Various functions to document SDynPy results in a PowerPoint document.  |
| [`sdynpy.doc.sdynpy_latex`](xref:api#sdynpy.doc.sdynpy_latex)  | [`sdynpy.doc.latex`](xref:api#sdynpy.doc.sdynpy_latex)     | Various functions to document SDynPy results in a LaTeX document.  |

## Demonstration Objects

Sometimes when a user has a new idea that they would like to try out, they simply want to quickly create a somewhat "interesting" system (e.g. something more than a simple beam or spring-mass-damper system), and generate data to explore.  The [`sdynpy.demo`](xref:api#sdynpy.demo) subpackage provides two demonstration objects: a flat plate and a simple airplane-like model made out of beam elements.  Note that there is some overhead in creating these demonstration objects, therefore the [`sdynpy.demo`](xref:api#sdynpy.demo) subpackage is not imported automatically with SDynPy, it must be imported separately using ``import sdynpy.demo``.  Because these are imported separately, there is no aliasing of the modules to the top-level namespace.

     
| Demonstration Modules                          | Alias                | Description                                      |
|---|---|---|
| [`sdynpy.demo.beam_airplane`](xref:api#sdynpy.demo.beam_airplane)  | None    | Creates a system and geometry for a structure that looks like a V-tailed airplane.  |
| [`sdynpy.demo.beam_plate`](xref:api#sdynpy.demo.beam_plate)  | None     | Creates a system and geometry for a structure that looks like a flat plate. |