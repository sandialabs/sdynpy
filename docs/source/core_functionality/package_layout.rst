Package Layout
================

This document will describe the layout of SDynPy to help users find the functions they are looking for.

SDynPy Subpackages
------------------

SDynPy is split up into several subpackages to help organize its content.  While these subpackages are convenient to organize the content, they are not convenient to use, as typical function calls might require descending multiple package levels, resulting in inconveniently long code.  Therefore, SDynPy heavily relies on aliases to commonly used functions and classes.

The following sections describe the major subpackages that exist in SDynPy.

SDynPy Core Objects
------------------------

The :py:mod:`sdynpy.core<sdynpy.core>` subpackage contains Python modules that define the core SDynPy objects and other core functionality.  These are designed to represent common data used in structural dynamics, including degrees of freedom, geometry, data, or dynamic systems.  Because of their ubiquitous nature, almost all of the modules, classes, and their corresponding helper functions are aliased to the top-level SDynPy namespace.  The :py:mod:`core<sdynpy.core>` modules are:

.. list-table::
    :widths: 25 25 50
    :header-rows: 1

    * - SDynPy Core Modules                            
      - Alias               
      - Description                                     
    * - :py:mod:`sdynpy.core.sdynpy_array<sdynpy.core.sdynpy_array>`       
      - :py:mod:`sdynpy.array<sdynpy.core.sdynpy_array>`      
      - Definition of the parent :py:class:`SdynpyArray<sdynpy.core.sdynpy_array.SdynpyArray>` class.   
    * - :py:mod:`sdynpy.core.sdynpy_colors<sdynpy.core.sdynpy_colors>`      
      - :py:mod:`sdynpy.colors<sdynpy.core.sdynpy_colors>`     
      - Definition of the color map scheme used by
        | SDynPy.    
    * - :py:mod:`sdynpy.core.sdynpy_coordinate<sdynpy.core.sdynpy_coordinate>`  
      - :py:mod:`sdynpy.coordinate<sdynpy.core.sdynpy_coordinate>` 
      - Definition of the :py:class:`CoordinateArray<sdynpy.core.sdynpy_coordinate.CoordinateArray>` class used to represent degrees of freedom.   
    * - :py:mod:`sdynpy.core.sdynpy_data<sdynpy.core.sdynpy_data>`        
      - :py:mod:`sdynpy.data<sdynpy.core.sdynpy_data>`       
      - Definition of the :py:class:`NDDataArray<sdynpy.core.sdynpy_data.NDDataArray>` class and it's subclasses. (:py:class:`TimeHistoryArray<sdynpy.core.sdynpy_data.TimeHistoryArray>`, :py:class:`SpectrumArray<sdynpy.core.sdynpy_data.SpectrumArray>`, etc.) used to represent data.
    * - :py:mod:`sdynpy.core.sdynpy_geometry<sdynpy.core.sdynpy_geometry>`    
      - :py:mod:`sdynpy.geometry<sdynpy.core.sdynpy_geometry>`   
      - Definition of the :py:class:`Geometry<sdynpy.core.sdynpy_geometry.Geometry>` class and its component classes (:py:class:`NodeArray<sdynpy.core.sdynpy_geometry.NodeArray>`, :py:class:`ElementArray<sdynpy.core.sdynpy_geometry.ElementArray>`, :py:class:`TracelineArray<sdynpy.core.sdynpy_geometry.TracelineArray>`, :py:class:`CoordinateSystemArray<sdynpy.core.sdynpy_geometry.CoordinateSystemArray>`) used to represent the locations and orientations of measurements.  
    * - :py:mod:`sdynpy.core.sdynpy_matrix<sdynpy.core.sdynpy_matrix>`      
      - :py:mod:`sdynpy.matrix_mod<sdynpy.core.sdynpy_matrix>` 
      - Definition of the :py:class:`Matrix<sdynpy.core.sdynpy_matrix.Matrix>` class used to map degrees of freedom to rows and columns of a matrix.   
    * - :py:mod:`sdynpy.core.sdynpy_shape<sdynpy.core.sdynpy_shape>`       
      - :py:mod:`sdynpy.shape<sdynpy.core.sdynpy_shape>`      
      - Definition of the parent :py:class:`ShapeArray<sdynpy.core.sdynpy_shape.ShapeArray>` class used to represent mode or deflection shapes.   
    * - :py:mod:`sdynpy.core.sdynpy_system<sdynpy.core.sdynpy_system>`      
      - :py:mod:`sdynpy.system<sdynpy.core.sdynpy_system>`     
      - Definition of the parent :py:class:`System<sdynpy.core.sdynpy_system.System>` class used to represent dynamic systems (mass, stiffness, and damping).   

In addition to the modules being aliased to the SDynPy namespace, many of the classes and functions from the :py:mod:`core<sdynpy.core>` subpackage are also aliased to the top-level namespace, as these are the most often called functions and classes in SDynPy.

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1

    * - SDynPy Array Class              
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_array.SdynpyArray<sdynpy.core.sdynpy_array.SdynpyArray>`       
      - :py:class:`sdynpy.SdynpyArray<sdynpy.core.sdynpy_array.SdynpyArray>`      
      - Parent class of all SDynPy Arrays.
     
.. list-table::
    :widths: 25 25 50
    :header-rows: 1
     
    * - Coordinate Classes and Functions 
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_coordinate.CoordinateArray<sdynpy.core.sdynpy_coordinate.CoordinateArray>`  
      - :py:class:`sdynpy.CoordinateArray<sdynpy.core.sdynpy_coordinate.CoordinateArray>` 
      - Class to represent degrees of freedom.   
    * - :py:func:`sdynpy.core.sdynpy_coordinate.coordinate_array<sdynpy.core.sdynpy_coordinate.coordinate_array>`  
      - :py:func:`sdynpy.coordinate_array<sdynpy.core.sdynpy_coordinate.coordinate_array>` 
      - Helper function to create :py:class:`CoordinateArray<sdynpy.core.sdynpy_coordinate.CoordinateArray>` objects.   

.. list-table::  
    :widths: 25 25 50
    :header-rows: 1
     
    * - Data Classes and Functions             
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_data.NDDataArray<sdynpy.core.sdynpy_data.NDDataArray>`        
      - :py:class:`sdynpy.NDDataArray<sdynpy.core.sdynpy_data.NDDataArray>`       
      - Parent class of all SDynPy Data Arrays. 
    * - :py:class:`sdynpy.core.sdynpy_data.TimeHistoryArray<sdynpy.core.sdynpy_data.TimeHistoryArray>`        
      - :py:class:`sdynpy.TimeHistoryArray<sdynpy.core.sdynpy_data.TimeHistoryArray>`       
      - Data class representing time histories. 
    * - :py:class:`sdynpy.core.sdynpy_data.TransferFunctionArray<sdynpy.core.sdynpy_data.TransferFunctionArray>`        
      - :py:class:`sdynpy.TransferFunctionArray<sdynpy.core.sdynpy_data.TransferFunctionArray>`       
      - Data class representing transfer functions or frequency response functions. 
    * - :py:class:`sdynpy.core.sdynpy_data.CoherenceArray<sdynpy.core.sdynpy_data.CoherenceArray>`        
      - :py:class:`sdynpy.CoherenceArray<sdynpy.core.sdynpy_data.CoherenceArray>`       
      - Data class representing coherence functions. 
    * - :py:class:`sdynpy.core.sdynpy_data.MultipleCoherenceArray<sdynpy.core.sdynpy_data.MultipleCoherenceArray>`        
      - :py:class:`sdynpy.MultipleCoherenceArray<sdynpy.core.sdynpy_data.MultipleCoherenceArray>`       
      - Data class representing multiple coherence functions. 
    * - :py:class:`sdynpy.core.sdynpy_data.PowerSpectralDensityArray<sdynpy.core.sdynpy_data.PowerSpectralDensityArray>`        
      - :py:class:`sdynpy.PowerSpectralDensityArray<sdynpy.core.sdynpy_data.PowerSpectralDensityArray>`       
      - Data class representing power spectral density functions. 
    * - :py:class:`sdynpy.core.sdynpy_data.SpectrumArray<sdynpy.core.sdynpy_data.SpectrumArray>`        
      - :py:class:`sdynpy.SpectrumArray<sdynpy.core.sdynpy_data.SpectrumArray>`       
      - Data class representing spectra (e.g. FFTs). 
    * - :py:class:`sdynpy.core.sdynpy_data.GUIPlot<sdynpy.core.sdynpy_data.GUIPlot>`        
      - :py:class:`sdynpy.GUIPlot<sdynpy.core.sdynpy_data.GUIPlot>`       
      - An interactive data plotter. 
    * - :py:class:`sdynpy.core.sdynpy_data.CPSDPlot<sdynpy.core.sdynpy_data.CPSDPlot>`        
      - :py:class:`sdynpy.CPSDPlot<sdynpy.core.sdynpy_data.CPSDPlot>`       
      - An interactive data plotter specifically for cross-power spectral density data. 
    * - :py:func:`sdynpy.core.sdynpy_data.data_array<sdynpy.core.sdynpy_data.data_array>`        
      - :py:func:`sdynpy.data_array<sdynpy.core.sdynpy_data.data_array>`       
      - Helper function to create :py:class:`NDDataArray<sdynpy.core.sdynpy_data.NDDataArray>` (and subclasses) objects. 
    * - :py:func:`sdynpy.core.sdynpy_data.time_history_array<sdynpy.core.sdynpy_data.time_history_array>`        
      - :py:func:`sdynpy.time_history_array<sdynpy.core.sdynpy_data.time_history_array>`       
      - Helper function to create :py:class:`TimeHistoryArray<sdynpy.core.sdynpy_data.TimeHistoryArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_data.transfer_function_array<sdynpy.core.sdynpy_data.transfer_function_array>`        
      - :py:func:`sdynpy.transfer_function_array<sdynpy.core.sdynpy_data.transfer_function_array>`       
      - Helper function to create :py:class:`TransferFunctionArray<sdynpy.core.sdynpy_data.TransferFunctionArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_data.coherence_array<sdynpy.core.sdynpy_data.coherence_array>`        
      - :py:func:`sdynpy.coherence_array<sdynpy.core.sdynpy_data.coherence_array>`       
      - Helper function to create :py:class:`CoherenceArray<sdynpy.core.sdynpy_data.CoherenceArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_data.multiple_coherence_array<sdynpy.core.sdynpy_data.multiple_coherence_array>`        
      - :py:func:`sdynpy.multiple_coherence_array<sdynpy.core.sdynpy_data.multiple_coherence_array>`       
      - Helper function to create :py:class:`MultipleCoherenceArray<sdynpy.core.sdynpy_data.MultipleCoherenceArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_data.power_spectral_density_array<sdynpy.core.sdynpy_data.power_spectral_density_array>`        
      - :py:func:`sdynpy.power_spectral_density_array<sdynpy.core.sdynpy_data.power_spectral_density_array>`       
      - Helper function to create :py:class:`PowerSpectralDensityArray<sdynpy.core.sdynpy_data.PowerSpectralDensityArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_data.spectrum_array<sdynpy.core.sdynpy_data.spectrum_array>`        
      - :py:func:`sdynpy.spectrum_array<sdynpy.core.sdynpy_data.spectrum_array>`       
      - Helper function to create :py:class:`SpectrumArray<sdynpy.core.sdynpy_data.SpectrumArray>` objects. 
     
.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - Geometry Classes and Functions 
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_geometry.Geometry<sdynpy.core.sdynpy_geometry.Geometry>`    
      - :py:class:`sdynpy.Geometry<sdynpy.core.sdynpy_geometry.Geometry>`   
      - Class to represent a test or analysis geometry.  
    * - :py:class:`sdynpy.core.sdynpy_geometry.NodeArray<sdynpy.core.sdynpy_geometry.NodeArray>`    
      - :py:class:`sdynpy.NodeArray<sdynpy.core.sdynpy_geometry.NodeArray>`   
      - Class to represent node locations in a geometry.  
    * - :py:class:`sdynpy.core.sdynpy_geometry.CoordinateSystemArray<sdynpy.core.sdynpy_geometry.CoordinateSystemArray>`    
      - :py:class:`sdynpy.CoordinateSystemArray<sdynpy.core.sdynpy_geometry.CoordinateSystemArray>`   
      - Class to represent global and local coordinate systems in a geometry.  
    * - :py:class:`sdynpy.core.sdynpy_geometry.TracelineArray<sdynpy.core.sdynpy_geometry.TracelineArray>`    
      - :py:class:`sdynpy.TracelineArray<sdynpy.core.sdynpy_geometry.TracelineArray>`   
      - Class to represent lines connecting nodes for visualization.  
    * - :py:class:`sdynpy.core.sdynpy_geometry.ElementArray<sdynpy.core.sdynpy_geometry.ElementArray>`    
      - :py:class:`sdynpy.ElementArray<sdynpy.core.sdynpy_geometry.ElementArray>`   
      - Class to represent elements connecting nodes for visualization.  
    * - :py:func:`sdynpy.core.sdynpy_geometry.node_array<sdynpy.core.sdynpy_geometry.node_array>`        
      - :py:func:`sdynpy.node_array<sdynpy.core.sdynpy_geometry.node_array>`       
      - Helper function to create :py:class:`NodeArray<sdynpy.core.sdynpy_geometry.NodeArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_geometry.coordinate_system_array<sdynpy.core.sdynpy_geometry.coordinate_system_array>`        
      - :py:func:`sdynpy.coordinate_system_array<sdynpy.core.sdynpy_geometry.coordinate_system_array>`       
      - Helper function to create :py:class:`CoordinateSystemArray<sdynpy.core.sdynpy_geometry.CoordinateSystemArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_geometry.traceline_array<sdynpy.core.sdynpy_geometry.traceline_array>`        
      - :py:func:`sdynpy.traceline_array<sdynpy.core.sdynpy_geometry.traceline_array>`       
      - Helper function to create :py:class:`TracelineArray<sdynpy.core.sdynpy_geometry.TracelineArray>` objects. 
    * - :py:func:`sdynpy.core.sdynpy_geometry.element_array<sdynpy.core.sdynpy_geometry.element_array>`        
      - :py:func:`sdynpy.element_array<sdynpy.core.sdynpy_geometry.element_array>`       
      - Helper function to create :py:class:`ElementArray<sdynpy.core.sdynpy_geometry.ElementArray>` objects. 
    * - :py:class:`sdynpy.core.sdynpy_geometry.id_map<sdynpy.core.sdynpy_geometry.id_map>`        
      - :py:class:`sdynpy.id_map<sdynpy.core.sdynpy_geometry.id_map>`       
      - Class to represent identification number maps between two geometries. 
     
.. list-table::   
    :widths: 25 25 50
    :header-rows: 1
     
    * - Matrix Classes and Functions    
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_matrix.Matrix<sdynpy.core.sdynpy_matrix.Matrix>`      
      - :py:class:`sdynpy.Matrix<sdynpy.core.sdynpy_matrix.Matrix>` 
      - Class used to map degrees of freedom to rows and columns of a matrix.   
    * - :py:func:`sdynpy.core.sdynpy_matrix.matrix<sdynpy.core.sdynpy_matrix.matrix>`      
      - :py:func:`sdynpy.matrix<sdynpy.core.sdynpy_matrix.matrix>` 
      - Helper function to create :py:class:`Matrix<sdynpy.core.sdynpy_matrix.Matrix>` objects.   
     
.. list-table::  
    :widths: 25 25 50
    :header-rows: 1
     
    * - Shape Classes and Functions       
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_shape.ShapeArray<sdynpy.core.sdynpy_shape.ShapeArray>`      
      - :py:class:`sdynpy.ShapeArray<sdynpy.core.sdynpy_shape.ShapeArray>` 
      - Class used to represent mode or deflection shapes.   
    * - :py:func:`sdynpy.core.sdynpy_shape.shape_array<sdynpy.core.sdynpy_shape.shape_array>`      
      - :py:func:`sdynpy.shape_array<sdynpy.core.sdynpy_shape.shape_array>` 
      - Helper function to create :py:class:`ShapeArray<sdynpy.core.sdynpy_shape.ShapeArray>` objects.   
     
.. list-table::    
    :widths: 25 25 50
    :header-rows: 1
     
    * - System Classes           
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.core.sdynpy_system.System<sdynpy.core.sdynpy_system.System>`      
      - :py:class:`sdynpy.System<sdynpy.core.sdynpy_system.System>`     
      - Class used to represent dynamic systems with mass, stiffness, and damping matrices.   


File Input and Output
------------------------

Structural dynamics data often comes from external sources, whether it is a modal, vibration, or shock test or an equivalent finite element simulation.  Therefore, being able to quickly and easily bring external data into SDynPy is a priority.  While users could strip data from their external files and manually construct SDynPy objects from that data, this risks bookkeeping and other translation errors.  Therefore if a file type is commonly read into or written from SDynPy, it is useful to add a dedicated reader or writer into SDynPy to handle this translation correctly.  The :py:mod:`sdynpy.fileio<sdynpy.fileio>` subpackage contains much of the code to handle these conversions.  For convenience, the modules are aliased to the top-level namespace.  The :py:mod:`fileio<sdynpy.fileio>` modules are:

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - File Input/Output Modules                        
      - Alias               
      - Description                                     
    * - :py:mod:`sdynpy.fileio.sdynpy_uff<sdynpy.fileio.sdynpy_uff>`       
      - :py:mod:`sdynpy.uff<sdynpy.fileio.sdynpy_uff>` or :py:mod:`sdynpy.unv<sdynpy.fileio.sdynpy_uff>`      
      - Functionality for reading and writing the `Universal File Format <https://www.ceas3.uc.edu/sdrluff/>`_, a text-based file format common in structural dynamics.   
    * - :py:mod:`sdynpy.fileio.sdynpy_rattlesnake<sdynpy.fileio.sdynpy_rattlesnake>` 
      - :py:mod:`sdynpy.rattlesnake<sdynpy.fileio.sdynpy_rattlesnake>` 
      - Functionality for reading output from the open-source vibration controller and modal testing software `Rattlesnake <https://github.com/sandialabs/rattlesnake-vibration-controller>`_ 
    * - :py:mod:`sdynpy.fileio.sdynpy_vic<sdynpy.fileio.sdynpy_vic>` 
      - :py:mod:`sdynpy.vic<sdynpy.fileio.sdynpy_vic>` 
      - Functionality for reading output from `Correlated Solution's VIC3D <https://www.correlatedsolutions.com/vic-3d>`_ ``.mat`` file output. 
    * - :py:mod:`sdynpy.fileio.sdynpy_pdf3D<sdynpy.fileio.sdynpy_pdf3D>` 
      - :py:mod:`sdynpy.pdf3D<sdynpy.fileio.sdynpy_pdf3D>` 
      - Functionality for writing geometry and shape data to a format that can be embedded into an `interactive PDF <https://helpx.adobe.com/acrobat/using/adding-3d-models-pdfs-acrobat.html>`_ for test or analysis documentation.  
    * - :py:mod:`sdynpy.fileio.sdynpy_tshaker<sdynpy.fileio.sdynpy_tshaker>` 
      - :py:mod:`sdynpy.tshaker<sdynpy.fileio.sdynpy_tshaker>` 
      - Functionality for reading output data from T-Shaker, a vibration shaker controller. 
    * - :py:mod:`sdynpy.fileio.sdynpy_escdf<sdynpy.fileio.sdynpy_escdf>` 
      - :py:mod:`sdynpy.escdf<sdynpy.fileio.sdynpy_escdf>` 
      - Functionality for reading and writing the Engineering Sciences Common Data Format. 

Finite Elements and Similar Numerical Functionality
----------------------------------------------------

SDynPy has a limited set of finite element and other numerical functionality in the :py:mod:`sdynpy.fem<sdynpy.fem>` subpackage.  This includes simple beam finite elements, electrodynamic shaker models, and sensor optimization routines to select optimal sensors for a test from finite element results.  For convenience, the modules are aliased to the top-level namespace.  The :py:mod:`fem<sdynpy.fem>` modules are:

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - FEM Modules                            
      - Alias               
      - Description                                     
    * - :py:mod:`sdynpy.fem.sdynpy_beam<sdynpy.fem.sdynpy_beam>`         
      - :py:mod:`sdynpy.beam<sdynpy.fem.sdynpy_beam>`      
      - Functionality for defining beam finite elements. 
    * - :py:mod:`sdynpy.fem.sdynpy_shaker<sdynpy.fem.sdynpy_shaker>`       
      - :py:mod:`sdynpy.shaker<sdynpy.fem.sdynpy_shaker>`    
      - Functionality for definining shaker electromechanical models per `Lang and Snyder <http://www.sandv.com/downloads/0110lang.pdf>`_.
    * - :py:mod:`sdynpy.fem.sdynpy_dof<sdynpy.fem.sdynpy_dof>`          
      - :py:mod:`sdynpy.dof<sdynpy.fem.sdynpy_dof>`       
      - Techniques such as effective independence used to select sensors for a test given finite element data 
    * - :py:mod:`sdynpy.fem.sdynpy_exodus<sdynpy.fem.sdynpy_exodus>`       
      - See Below          
      - Functionality for reading and writing the `Exodus <https://sandialabs.github.io/seacas-docs/exodusII-new.pdf>`_ finite element model format. 


Because the Exodus file format is used often at Sandia National Laboratories where SDynPy was originally developed, key classes from the :py:mod:`sdynpy_exodus<sdynpy.fem.sdynpy_exodus>` module are also aliased to the top-level namespace.

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - Exodus Classes      
      - Alias               
      - Description                                     
    * - :py:class:`sdynpy.fem.sdynpy_exodus.Exodus<sdynpy.fem.sdynpy_exodus.Exodus>`      
      - :py:class:`sdynpy.Exodus<sdynpy.fem.sdynpy_exodus.Exodus>` 
      - Class to represent an Exodus file as stored on the filesystem   
    * - :py:class:`sdynpy.fem.sdynpy_exodus.ExodusInMemory<sdynpy.fem.sdynpy_exodus.ExodusInMemory>`      
      - :py:class:`sdynpy.ExodusInMemory<sdynpy.fem.sdynpy_exodus.ExodusInMemory>` 
      - Class that represents an Exodus file in memory in a format similar to a Matlab Structure   

Modal Analysis
------------------------

SDynPy has capabilities for performing experimental modal analysis which entails fitting modes to frequency response functions measured on the test article.  These and similar capabilities exist in the :py:mod:`sdynpy.modal<sdynpy.modal>` subpackage.  Many of the modules in this subpackage provide both code-based and GUI-based tools to fit modes to data.  In the case of the :py:mod:`modal<sdynpy.modal>` subpackage, much of the useful content boils down to a handful of classes.  Therefore, instead of aliasing the modules in the subpackage to the top-level namespace, SDynPy aliases these classes to the top-level namespace.

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - Modal Classes
      - Alias
      - Description                                     
    * - :py:class:`sdynpy.modal.sdynpy_polypy.PolyPy<sdynpy.modal.sdynpy_polypy.PolyPy>` 
      - :py:class:`sdynpy.PolyPy<sdynpy.modal.sdynpy_polypy.PolyPy>`    
      - Code-based implementation of the PolyMax mode fitter. 
    * - :py:class:`sdynpy.modal.sdynpy_polypy.PolyPy_GUI<sdynpy.modal.sdynpy_polypy.PolyPy_GUI>` 
      - :py:class:`sdynpy.PolyPy_GUI<sdynpy.modal.sdynpy_polypy.PolyPy_GUI>`    
      - GUI-based implementation of the PolyMax mode fitter. 
    * - :py:class:`sdynpy.modal.sdynpy_smac.SMAC<sdynpy.modal.sdynpy_smac.SMAC>` 
      - :py:class:`sdynpy.SMAC<sdynpy.modal.sdynpy_smac.SMAC>` 
      - Code-based implemenation of the Synthesize Modes and Correlate mode fitter. 
    * - :py:class:`sdynpy.modal.sdynpy_smac.SMAC_GUI<sdynpy.modal.sdynpy_smac.SMAC_GUI>` 
      - :py:class:`sdynpy.SMAC_GUI<sdynpy.modal.sdynpy_smac.SMAC_GUI>` 
      - GUI-based implemenation of the Synthesize Modes and Correlate mode fitter. 
    * - :py:class:`sdynpy.modal.sdynpy_ccmif.ColoredCMIF<sdynpy.modal.sdynpy_ccmif.ColoredCMIF>` 
      - :py:class:`sdynpy.ColoredCMIF<sdynpy.modal.sdynpy_ccmif.ColoredCMIF>` 
      - GUI-based tool to interactively select and combine modes fit to multiple single-reference measurements. 
    * - :py:class:`sdynpy.modal.sdynpy_signal_processing_gui.SignalProcessingGUI<sdynpy.modal.sdynpy_signal_processing_gui.SignalProcessingGUI>` 
      - :py:class:`sdynpy.SignalProcessingGUI<sdynpy.modal.sdynpy_signal_processing_gui.SignalProcessingGUI>` 
      - GUI-based tool to compute spectral quantities from time histories. 
    * - :py:class:`sdynpy.modal.sdynpy_modal_test.ModalTest<sdynpy.modal.sdynpy_modal_test.ModalTest>` 
      - :py:class:`sdynpy.ModalTest<sdynpy.modal.sdynpy_modal_test.ModalTest>` 
      - A class to represent a typical modal test, intended to aid modal test practitioners in the data processing and documentation workflow. 

Signal Processing
------------------------

The :py:mod:`sdynpy.signal_processing<sdynpy.signal_processing>` subpackage includes a wide variety of tools that are related to structural dynamics.  Often, the code in this package is at a "lower level" than the rest of the SDynPy code, operating on raw data rather than SDynPy objects.  Many times, SDynPy objects will often call these lower-level functions as part of their own operations.  :py:mod:`signal_processing<sdynpy.signal_processing>` functions can be faster in that they remove the bookkeeping overhead that a lot of SDynPy objects implement, but the user must then be aware that their data must be correctly sorted prior to using the techniques.  Additionally, some of the content in :py:mod:`signal_processing<sdynpy.signal_processing>` does not fit well within the existing SDynPy core objects, so the subpackage serves as a kind of catch-all for tools that can be used in structural dynamics, but perhaps are not commonly used.  The modules of :py:mod:`signal_processing<sdynpy.signal_processing>` are generally aliased to the top-level namespace.

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - Signal Processing Modules                            
      - Alias               
      - Description                                     
    * - :py:mod:`sdynpy.signal_processing.sdynpy_camera<sdynpy.signal_processing.sdynpy_camera>` 
      - :py:mod:`sdynpy.camera<sdynpy.signal_processing.sdynpy_camera>`    
      - Various functions for dealing with data from cameras, including camera calibration. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_complex<sdynpy.signal_processing.sdynpy_complex>` 
      - :py:mod:`sdynpy.complex<sdynpy.signal_processing.sdynpy_complex>`    
      - Utility functions for working with complex numbers. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_correlation<sdynpy.signal_processing.sdynpy_correlation>` 
      - :py:mod:`sdynpy.correlation<sdynpy.signal_processing.sdynpy_correlation>`    
      - Functions like the modal assurance cirterion for comparing data. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_cpsd<sdynpy.signal_processing.sdynpy_cpsd>` 
      - :py:mod:`sdynpy.cpsd<sdynpy.signal_processing.sdynpy_cpsd>`    
      - Various functions for dealing with cross-power spectral density matrices. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_frf<sdynpy.signal_processing.sdynpy_frf>` 
      - :py:mod:`sdynpy.frf<sdynpy.signal_processing.sdynpy_frf>`    
      - Various functions computing frequency response functions. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_frf_inverse<sdynpy.signal_processing.sdynpy_frf_inverse>` 
      - :py:mod:`sdynpy.frf_inverse<sdynpy.signal_processing.sdynpy_frf_inverse>`    
      - Various functions for inverting frequency response functions, including regularization. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_generator<sdynpy.signal_processing.sdynpy_generator>` 
      - :py:mod:`sdynpy.generator<sdynpy.signal_processing.sdynpy_generator>`    
      - Various functions for generating common time signals, like pseudorandom or sine sweeps. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_geometry_fitting<sdynpy.signal_processing.sdynpy_geometry_fitting>` 
      - :py:mod:`sdynpy.geometry_fitting<sdynpy.signal_processing.sdynpy_geometry_fitting>`    
      - Various functions for dealing with geometry; for example, fitting a shape to a point cloud, or finding intersection points of lines. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_harmonic<sdynpy.signal_processing.sdynpy_harmonic>` 
      - :py:mod:`sdynpy.harmonic<sdynpy.signal_processing.sdynpy_harmonic>`    
      - Functions for fitting sine waves to data. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_integration<sdynpy.signal_processing.sdynpy_integration>` 
      - :py:mod:`sdynpy.integration<sdynpy.signal_processing.sdynpy_integration>`    
      - Various functions for generating and integrating state space systems. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_lrm<sdynpy.signal_processing.sdynpy_lrm>` 
      - :py:mod:`sdynpy.lrm<sdynpy.signal_processing.sdynpy_lrm>`    
      - Advanced FRF computation technique using local rational modeling. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_rotation<sdynpy.signal_processing.sdynpy_rotation>` 
      - :py:mod:`sdynpy.rotation<sdynpy.signal_processing.sdynpy_rotation>`    
      - Various functions for dealing with rotations; rotation matrices, Rodrigues parameters, rigid body transformations. 
    * - :py:mod:`sdynpy.signal_processing.sdynpy_srs<sdynpy.signal_processing.sdynpy_srs>` 
      - :py:mod:`sdynpy.srs<sdynpy.signal_processing.sdynpy_srs>`    
      - Various functions for working with shock response spectra. 

Documentation
------------------------

SDynPy has limited functionality to create documentation for test or analysis results.  This functionality is currently a work-in-progress, but capability exists to dump common formats into PowerPoint or LaTeX documents.

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - Documentation Modules                               
      - Alias               
      - Description                                     
    * - :py:mod:`sdynpy.doc.sdynpy_ppt<sdynpy.doc.sdynpy_ppt>` 
      - :py:mod:`sdynpy.doc.ppt<sdynpy.doc.sdynpy_ppt>`    
      - Various functions to document SDynPy results in a PowerPoint document. 
    * - :py:mod:`sdynpy.doc.sdynpy_latex<sdynpy.doc.sdynpy_latex>` 
      - :py:mod:`sdynpy.doc.latex<sdynpy.doc.sdynpy_latex>`    
      - Various functions to document SDynPy results in a LaTeX document. 

Demonstration Objects
------------------------

Sometimes when a user has a new idea that they would like to try out, they simply want to quickly create a somewhat "interesting" system (e.g. something more than a simple beam or spring-mass-damper system), and generate data to explore.  The :py:mod:`sdynpy.demo<sdynpy.demo>` subpackage provides two demonstration objects: a flat plate and a simple airplane-like model made out of beam elements.  Note that there is some overhead in creating these demonstration objects, therefore the :py:mod:`sdynpy.demo<sdynpy.demo>` subpackage is not imported automatically with SDynPy, it must be imported separately using ``import sdynpy.demo``.  Because these are imported separately, there is no aliasing of the modules to the top-level namespace.

.. list-table:: 
    :widths: 25 25 50
    :header-rows: 1
     
    * - Demonstration Modules                         
      - Alias               
      - Description                                     
    * - :py:mod:`sdynpy.demo.beam_airplane<sdynpy.demo.beam_airplane>` 
      - None   
      - Creates a system and geometry for a structure that looks like a V-tailed airplane. 
    * - :py:mod:`sdynpy.demo.beam_plate<sdynpy.demo.beam_plate>` 
      - None    
      - Creates a system and geometry for a structure that looks like a flat plate.