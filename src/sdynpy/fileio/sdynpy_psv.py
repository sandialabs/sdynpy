# -*- coding: utf-8 -*-
"""
Load and wrap data from Polytec PSV files.

This module provides a Python wrapper around the Polytec ``Polyfile`` COM
interface, along with convenience methods for extracting common data products
into ``sdynpy`` objects.

Notes
-----
- This module requires ``pywin32`` and therefore only works on Windows.
- Many objects exposed by this module are thin wrappers around COM objects.
- Where the exact runtime type returned by COM is uncertain, conservative type
  annotations such as ``Any`` are used.
"""

from __future__ import annotations

"""
Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


from enum import Enum, IntFlag
from io import BytesIO
from os import PathLike
from typing import Any, TypeVar
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

try:
    import win32com.client as client
    from pywintypes import com_error
except ModuleNotFoundError:
    client = None
    # Fallback keeps ``except com_error`` clauses valid even when pywin32
    # is unavailable. Those paths are normally unreachable because PolyFile
    # raises during initialization when ``client is None``.
    com_error = RuntimeError

from sdynpy.core.sdynpy_coordinate import coordinate_array
from sdynpy.core.sdynpy_data import (
    coherence_array,
    multiple_coherence_array,
    power_spectral_density_array,
    time_history_array,
    transfer_function_array,
    CoherenceArray,
    MultipleCoherenceArray,
    PowerSpectralDensityArray,
    TimeHistoryArray,
    TransferFunctionArray,
)
from sdynpy.core.sdynpy_geometry import (
    Geometry,
    coordinate_system_array,
    node_array,
)
from sdynpy.signal_processing.sdynpy_geometry_fitting import intersection_point_multiple_lines, distance_point_line

VERBOSE = False

_FlagT = TypeVar("_FlagT", bound=IntFlag)


class AcquisitionMode(Enum):
    """Measurement acquisition mode."""

    TIME = 0
    FFT = 1
    ZOOM_FFT = 2
    MULTI_FRAME = 3
    FAST_SCAN = 4
    MODE_ORDER_TRACKING = 5
    IQ = 6
    IN_PLANE = 7


class AverageType(Enum):
    """Averaging mode."""

    OFF = 0
    MAGNITUDE = 1
    COMPLEX = 2
    TIME = 3
    PEAK_HOLD = 4


class DigitalFilterQuality(Enum):
    """Quality setting for digital filters."""

    VERY_LOW = 0
    LOW = 1
    MIDDLE = 2
    HIGH = 3
    VERY_HIGH = 4
    VERY_HIGH_200 = 5
    VERY_HIGH_300 = 6
    VERY_HIGH_400 = 7
    VERY_HIGH_500 = 8


class DigitalFilterType(Enum):
    """Digital filter type."""

    LOWPASS = 0
    HIGHPASS = 1
    BANDPASS = 2
    NOTCH = 3
    NONE = 4


class Direction(Enum):
    """Measurement direction."""

    Z_POS = 0
    Z_NEG = 1
    X_POS = 2
    X_NEG = 3
    Y_POS = 4
    Y_NEG = 5
    VECTOR_3D = 6


class Coupling(Enum):
    """Input coupling mode."""

    AC = 0
    DC = 1
    UNKNOWN = 2


class PhysicalQuantity(Enum):
    """Physical quantity identifier."""

    ACCELERATION = 0
    ANGLE = 1
    ANGULAR_ACCELERATION = 2
    ANGULAR_VELOCITY = 3
    DISPLACEMENT = 4
    ELECTRICAL_CURRENT = 5
    FORCE = 6
    POWER = 7
    PRESSURE = 8
    REVOLUTION = 9
    RPM = 10
    SOUND_PRESSURE = 11
    TORQUE = 12
    VELOCITY = 13
    VOLTAGE = 14
    COUNTS = 15
    VOLUME = 16
    VOLUME_VELOCITY = 17
    VOLUME_ACCELERATION = 18
    STRAIN = 19
    GRAVITY = 20
    STRESS = 21
    TIME = 22
    FREQUENCY = 23
    ANGLE_DEGREE = 24
    ANGULAR_ACCELERATION_DEGREE = 25
    ANGULAR_VELOCITY_DEGREE = 26


class ChannelType(Enum):
    """Acquisition channel type."""

    ANALOG = 0
    ANALOG_FIXED = 1
    FRINGE_COUNTER = 2
    LOGICAL = 3
    MODULATED = 4
    DIGITAL_VIB = 5
    MULTI_POINT_VIB = 6
    LOOP_BACK = 7
    ANALOG_HF = 8


class WindowFunction(Enum):
    """Window function used for time/FFT processing."""

    RECTANGLE = 0
    BARTLETT = 1
    BLACKMANHARRIS = 2
    EXPONENTIAL = 3
    FLATTOP = 4
    FORCE = 5
    HAMMING = 6
    HANNING = 7
    TAPEREDHANNING = 8


class BandwidthExtension(Enum):
    """Bandwidth-extension setting."""

    NONE = 0
    OFF = 1
    ON1 = 2
    ON2 = 3


class WaveformType(Enum):
    """Signal-generator waveform type."""

    SINE = 0
    SQUARE = 1
    TRIANGLE = 2
    RAMP = 3
    SWEEP = 4
    BURST_CHIRP = 5
    BURST_RANDOM = 6
    PERIODIC_CHIRP = 7
    PSEUDORANDOM = 8
    WHITE_NOISE = 9
    USER_DEFINED = 10
    PULSE = 11
    MULTI_CARRIER_CW = 12


class MultiFrameMode(Enum):
    """Multi-frame acquisition mode."""

    AUTOMATIC = 0
    REGULAR = 1
    MANUAL = 2


class SignalEnhancementMode(Enum):
    """Signal enhancement (SE) mode."""

    FAST = 0
    FAST_STANDARD = 1
    STANDARD = 2
    STANDARD_BEST = 3
    BEST = 4


class TriggerEdge(Enum):
    """Trigger edge polarity."""

    RISING = 0
    FALLING = 1


class TriggerSource(Enum):
    """Trigger source."""

    OFF = 0
    EXTERNAL = 1
    ANALOG = 2
    INTERNAL = 3


class ControllerCaps(IntFlag):
    """Controller capability flags."""

    NONE = 0
    POWER_UP = 1
    SENSOR_HEAD = 2
    SERVICE_MODE = 4
    DIGITAL_DEMODULATION = 8
    MULTIPLE_OVERRANGE = 16
    SIGNAL_ENHANCEMENT = 32
    MEASUREMENT_MODE = 64
    MULTIPLE_SENSOR_HEADS = 128
    OVERRANGE = 256
    PSV500_BEHAVIOUR = 512
    NO_DECODER = 1024
    DECODER_CLEAR_MODE = 2048
    IS_PHYSICAL_UNIT = 4096
    WRITE_LOCK = 8192
    RECORDER = 16384
    WEB_SERVER = 32768


def decode_controller_caps(caps_value: int) -> ControllerCaps:
    """Decode an integer bitmask into :class:`ControllerCaps`."""
    return ControllerCaps(int(caps_value))


def controller_caps_list(caps_value: int) -> list[ControllerCaps]:
    """Return individual controller capability flags set in ``caps_value``."""
    caps = decode_controller_caps(caps_value)
    return [flag for flag in ControllerCaps if flag is not ControllerCaps.NONE and (caps & flag)]


class SensorHeadCaps(IntFlag):
    """Sensor-head capability flags."""

    NONE = 0
    FOCUS_RELATIVE = 1
    FOCUS_ABSOLUTE = 2
    AUTO_FOCUS = 4
    MANUAL_FOCUS_ENABLE = 8
    SHUTTER = 16
    DIMMER = 32
    SIGNAL_LEVEL = 64
    VIDEO_CAMERA = 128
    SIGNAL_BALANCE = 256
    FOCUS_LIMIT = 512
    FOCUS_PROGRESS = 1024
    BRAGG_CELL_FREQUENCY = 2048
    ILLUMINATION_STATE = 4096
    ILLUMINATION_INTENSITY = 8192
    AUTO_FOCUS_IN_RANGE = 16384
    AUTO_FOCUS_IN_AREA = 32768
    LASER_DELAY = 65536
    SIGNAL_LEVEL_B = 131072
    BEAM_GAP = 262144
    STANDOFF_DIST = 524288
    SENSITIVITY = 1048576
    RETICLE = 2097152
    OVER_TEMP = 4194304
    TEMPERATURE = 8388608
    COHERENCE_OPTIMIZER = 16777216
    ALIGNMENT_LASER_SHUTTER = 33554432
    OBJECTIVE_TEMPERATURE = 67108864
    FOCUS_STEP_DEVIATION = 134217728
    LASER_WAVELENGTH = 268435456
    LASER_WAVELENGTH_STABILIZATION = -2147483648
    PILOT_LASER_SHUTTER = 536870912
    PILOT_LASER_DIMMER = 1073741824


class SensorHeadCaps2(IntFlag):
    """Extended sensor-head capability flags."""

    NONE = 0
    LASER_TEMPERATURE = 1
    LASER_TEMPERATURE_OUT_OF_RANGE = 2
    SONDE = 4
    UNIQUE_ID = 8
    RSSI_LEVEL_BLINK = 16
    RGB_LED_COLOR = 32
    LASER_POWER = 64
    DIFFERENTIAL_MEASUREMENT_MODE = 128
    EXTERNAL_SCANNER_CONTROL = 256
    LASER_POWER_SUPPLY = 512
    Q_TEC = 1024


class SignalBalanceState(Enum):
    """Signal-balance state."""

    SEEK = 0
    MINUS3 = 1
    MINUS2 = 2
    MINUS1 = 3
    BALANCED = 4
    PLUS1 = 5
    PLUS2 = 6
    PLUS3 = 7


class FilterType(Enum):
    """Filter type used in controller/filter settings."""

    LOW_PASS = 0
    HIGH_PASS = 1
    ADAPTIVE = 2
    TRACKING = 3
    DC_RESPONSE = 4
    TRANSLATION = 5
    ROTATION = 6
    RPM_RESPONSE = 7
    ASE = 8
    DISPLACEMENT_LOW_PASS = 9
    DISPLACEMENT_HIGH_PASS = 10
    ACCELERATION_LOW_PASS = 11
    ACCELERATION_HIGH_PASS = 12
    DISPLACEMENT_ASE = 13
    ACCELERATION_ASE = 14


class QuantityType(Enum):
    """Quantity type used in controller quantity settings."""

    ACCELERATION = 0
    ANGLE = 1
    ANGULAR_ACCELERATION = 2
    ANGULAR_VELOCITY = 3
    CHARGE = 4
    COUNTS = 5
    CURRENT = 6
    DISPLACEMENT = 7
    FORCE = 8
    FREQUENCY = 9
    LUMINOUS_INTENSITY = 10
    MASS = 11
    ORDER = 12
    POWER = 13
    PRESSURE = 14
    RESISTANCE = 15
    REVOLUTION = 16
    RPM = 17
    SOUND_PRESSURE = 18
    SUBSTANCE = 19
    TEMPERATURE = 20
    TIME = 21
    TORQUE = 22
    USER = 23
    VELOCITY = 24
    VOLTAGE = 25
    WORK = 26
    VOLUME = 27
    VOLUME_VELOCITY = 28
    VOLUME_ACCELERATION = 29
    STRAIN = 30
    GRAVITATIONAL_ACCELERATION = 31
    STRESS = 32
    ANGLE_DEGREE = 33
    ANGULAR_ACCELERATION_DEGREE = 34
    ANGULAR_VELOCITY_DEGREE = 35


class AlignmentPoint3DCaps(IntFlag):
    """3D alignment-point capability flags."""

    COORD_3D = 1
    DISTANCE = 2
    QUALITY = 4
    SCANNER = 8
    VIDEO = 16


class AlignmentPoint3DType(Enum):
    """3D alignment-point type."""

    ALIGN = 0
    OBJ_FREE = 1
    OBJ_ORIGIN = 2
    OBJ_POS_X_AXIS = 3
    OBJ_POS_Y_AXIS = 4
    OBJ_POS_Z_AXIS = 5
    OBJ_X_POS_Y_PLANE = 6
    OBJ_X_POS_Z_PLANE = 7
    OBJ_Y_POS_X_PLANE = 8
    OBJ_Y_POS_Z_PLANE = 9
    OBJ_Z_POS_X_PLANE = 10
    OBJ_Z_POS_Y_PLANE = 11
    AUTO_ALIGN = 12


class CoordDefinitionMode(Enum):
    """3D coordinate-definition mode."""

    OFF = 0
    ORIG_POINT_ON_AXIS_POINT_ON_PLANE = 1
    THREE_POINTS_ON_AXES = 2
    FREE_POINTS = 3


class ScanHeadType(Enum):
    """Scanning-head type."""

    NONE = 0
    OFV040 = 1
    OFV048_DOWN = 2
    OFV048_FRONT = 3
    OFV050 = 4
    OFV055 = 5
    OFV055F = 6
    OFV056 = 7
    MSV = 8
    VPI = 9
    PSVI400_MR = 10
    PSVI400_LR = 11
    MSV_NTSC = 12
    MSAI400 = 13
    MSAI500 = 14
    UHF120 = 15
    OFV534 = 16
    PSV500 = 17
    MSAI100 = 18
    MPVI800 = 19
    MSAI600 = 20
    MSAI620 = 21
    MSAI650 = 22
    PSV6XX = 23
    MSAI630 = 24
    VIO_I_130 = 25
    VFX_I_130 = 26
    PSVI7XX = 27


class ScanStatus(IntFlag):
    """Scan-point status flags."""

    NONE = 0
    VALID = 1
    OPTIMAL = 2
    OVERRANGE = 4
    INVALIDATED = 8
    DISABLED = 16
    INVALID_FRAMES = 32
    NOT_REACHABLE = 64
    HIDDEN = 128
    VIDEO_TRIANGULATION_FAILED = 256
    INTERPOLATE = 512
    INTERPOLATION_FAILED = 1024
    ASSIGNED_CHANNEL_1D = 2048
    ASSIGNED_CHANNEL_3D = 4096
    ASSIGNED_SENSOR = 8192


class FrontEndCaps(IntFlag):
    """Front-end capability flags."""

    CONTROLLABLE = 1
    DIFFERENTIAL_INPUT = 2
    FE_3D = 4
    MULTIPLE_SCAN_HEAD = 8
    VDD = 16
    FRINGE_COUNTER = 32
    S_VIDEO = 64
    IO_FRONT_END = 128
    FIXED_VIB_CHANNELS = 256
    REFERENCE_CHANNELS = 512
    DC_COUPLING_FOR_VIB_CHANNELS = 1024
    IEPE_ENFORCE_AC_COUPLING = 2048
    DIGITAL_VIDEO = 4096
    MULTI_POINT_VIB = 8192
    NO_HW_DECODER = 16384
    SINGLE_POINT_3D = 32768


class AcqBoard(Enum):
    """Acquisition-board type."""

    UNKNOWN = 0
    NONE = 1
    PC430 = 2
    SP216 = 3
    NI4451 = 4
    NI4452 = 5
    NI6110 = 6
    NI6111 = 7
    NI6110_20MHZ = 8
    NI6111_20MHZ = 9
    NI6601 = 10
    NI6602 = 11
    NI6601_20MHZ = 12
    NI6602_20MHZ = 13
    SIMULATION = 14
    NI6534 = 15
    NI6534_20MHZ = 16
    SPDIF = 17
    MI3025 = 18
    NI6115 = 19
    NI6115_20MHZ = 20
    NI4461 = 21
    NI4462 = 22
    VIB_E_220 = 23
    LECROY_WR204XI = 24
    M2I3027 = 25
    LECROY_WP725ZI = 26
    POLY_ETH = 27
    LECROY_WP725ZI_A = 28
    M2I4961 = 29
    MPV800 = 30
    LECROY_WR8254M = 31
    ADQ7 = 32
    M4I4450 = 33
    M2P5913 = 34
    M2P5943 = 35
    M2P5912 = 36
    M2P5942 = 37
    ROHDE_SCHWARZ_RTP = 38


class FrontEndType(Enum):
    """Front-end type."""

    UNKNOWN = 0
    NONE = 1
    BNC2140 = 2
    PSV_Z_010 = 3
    PSV_Z_040 = 4
    PSV_Z_040F = 5
    VIB_Z_010 = 6
    VDD_Z_010 = 7
    MSV_Z_040 = 8
    VIB_Z_014 = 9
    VIB_Z_015 = 10
    VDD_Z_011 = 11
    VIB_Z_016 = 12
    VIB_Z_012 = 13
    VIB_Z_017 = 14
    PSV_Z_040H = 15
    PSV_Z_040U = 16
    PSV_Z_3D_H = 17
    PSV_Z_1D_H = 18
    MSA_E_400_M2 = 19
    MSA_E_400_M4 = 20
    MSA_E_400_VDD = 21
    PSV_E_400_B = 22
    PSV_E_400_H4 = 23
    PSV_E_400_M2 = 24
    PSV_E_400_M2_20 = 25
    PSV_E_400_M4 = 26
    PSV_E_400_3D = 27
    PSV_E_400_3D_1D = 28
    MSA_E_400_M2_20 = 29
    PSV_E_400_3D_M4 = 30
    PSV_E_400_3D_1D_M4 = 31
    MSA_E_401_M2 = 32
    MSA_E_401_M4 = 33
    MSA_E_401_M2_20 = 34
    MSA_E_401_VDD = 35
    VDD_E_600 = 36
    PSV_E_401_B = 37
    PSV_E_401_H4 = 38
    PSV_E_401_M2 = 39
    PSV_E_401_M2_20 = 40
    PSV_E_401_M4 = 41
    PSV_E_401_3D = 42
    PSV_E_401_3D_1D = 43
    PSV_E_401_3D_M4 = 44
    PSV_E_401_3D_1D_M4 = 45
    MSA_E_401_4_M4 = 46
    PSV_E_400_1D = 47
    PSV_E_400_1D_M4 = 48
    PSV_E_401_1D = 49
    PSV_E_401_1D_M4 = 50
    PCI4461 = 51
    PCI4462 = 52
    VIB_E_220 = 53
    MSA_E_500_M2 = 54
    MSA_E_500_M4 = 55
    MSA_E_500_M2_20 = 56
    MSA_E_500_VDD = 57
    MSA_E_500_4_M4 = 58
    LECROY_WR204XI = 60
    LECROY_WP725ZI = 61
    VIB_E_400_80 = 62
    VIB_E_400_84 = 63
    VIB_E_400_1000 = 64
    VIB_E_400_1004 = 65
    VIB_E_400_VDD = 66
    VIB_E_400_3D_84 = 67
    VIB_E_400_3D_1004 = 68
    PSV_E_400_1D_M2_20 = 69
    PSV_E_401_1D_M2_20 = 70
    PSV_E_401_VDD = 71
    PSV_E_401_2_M4 = 72
    MSA_E_050_3D_1D_84 = 74
    MSA_E_050_1D_84 = 75
    MSA_E_050_3D_1004 = 76
    MSA_E_050_3D_1D_1004 = 77
    MSA_E_050_1D_1004 = 78
    PSV_F_500 = 79
    PSV_F_500_M = 80
    PSV_F_500_3D = 81
    PSV_F_500_3D_1D = 82
    PSV_F_500_1D = 83
    PSV_F_500_3D_M = 84
    PSV_F_500_3D_1D_M = 85
    PSV_F_500_1D_M = 86
    LECROY_WP725ZI_A = 87
    PSV_F_500_V = 88
    PSV_F_500_3D_V = 89
    PSV_F_500_3D_1D_V = 90
    PSV_F_500_1D_V = 91
    MSA_F_100_3D = 92
    MSA_F_100_3D_1D = 93
    MSA_F_100_1D = 94
    MSA_F_100_3D_M = 95
    MSA_F_100_3D_1D_M = 96
    MSA_F_100_1D_M = 97
    MSA_F_100_3D_V = 98
    MSA_F_100_3D_1D_V = 99
    MSA_F_100_1D_V = 100
    PSV_F_500_B = 101
    VIBSOFT_M2_40 = 102
    VIBSOFT_M4_25 = 103
    TMS_350 = 104
    MPV_E_800 = 105
    MSA_E_050_80 = 106
    MSA_E_050_84 = 107
    MSA_E_050_1000 = 108
    MSA_E_050_1004 = 109
    MSA_E_050_M2_20 = 110
    MSA_F_600_V = 111
    LECROY_WR8254M = 112
    MSA_F_600_M = 113
    VFX = 114
    IVS500 = 115
    MSA_F_620 = 116
    VGO200 = 117
    VIO130 = 118
    M4I4450 = 119
    MSA_F_650_M = 120
    MSA_F_650_V = 121
    VIBSOFT_M8_25 = 122
    PSV_F_600 = 123
    PSV_F_600_3D = 124
    PSV_F_600_3D_1D = 125
    PSV_F_600_1D = 126
    PSV_F_600_V = 127
    PSV_F_600_3D_V = 128
    PSV_F_600_3D_1D_V = 129
    PSV_F_600_1D_V = 130
    VGO200_P = 131
    MSA_F_630 = 132
    VIBSOFT_PRO = 133
    VIBSOFT_PRO_05 = 134
    VIBSOFT_PRO_3 = 135
    VIBSOFT_PRO_24 = 136
    VIBSOFT_PRO_3D = 137
    VIBSOFT_PRO_05_3D = 138
    VIBSOFT_PRO_3_3D = 139
    VIBSOFT_PRO_24_3D = 140
    MSA_E_060_PRO = 141
    MSA_E_060_PRO_05 = 142
    MSA_E_060_PRO_3 = 143
    MSA_E_060_PRO_24 = 144
    PSV_F_700 = 145
    PSV_F_700_3D = 146
    PSV_F_700_3D_1D = 147
    PSV_F_700_1D = 148
    PSV_I_780 = 149
    PSV_I_730 = 150
    VIBSOFT_PRO_3D_1D = 151
    VIBSOFT_PRO_05_3D_1D = 152
    VIBSOFT_PRO_3_3D_1D = 153
    VIBSOFT_PRO_24_3D_1D = 154


class GeneratorType(Enum):
    """Signal-generator type."""

    UNKNOWN = 0
    NONE = 1
    SIMULATION = 2
    HP33120A = 3
    HP33250A = 4
    PREMA1000 = 5
    NI4451 = 6
    NI611X = 7
    NI671X = 8
    MI60XX = 9
    NI673X = 10
    NI4461 = 11
    M2I60XX = 12
    POLY_ETHERNET = 13
    RS_SMBV100A = 14
    MPV800 = 15
    M4I6630 = 16
    M2P6570 = 17
    PSV700 = 18


class GeometryStatus(IntFlag):
    """3D geometry status flags."""

    NONE = 0
    VALID = 1
    MEASURED = 4
    IMPORTED = 8
    TRIANGULATION = 16
    INTERPOLATED = 32
    MODIFIED = 64
    TOO_BRIGHT = 128
    TOO_DARK = 256
    FAILED = 512
    OPTIMAL_MEASURED = 1024
    CALCULATED_3D_FROM_2D = 2048
    CALCULATED_2D_FROM_3D = 4096
    VIDEO_TRIANGULATED = 8192
    CALCULATED_3D_FROM_2D_AND_DISTANCE = 16384


class FocusStatus(IntFlag):
    """Laser-focus status flags."""

    NONE = 0
    VALID = 1
    MANUAL = 2
    FAST = 4
    BEST = 8
    INTERPOLATED = 16
    FAILED = 32
    CALCULATED = 64
    AUTO = 128


class MeasurementLocationMode(Enum):
    """Measurement-location mode."""

    FIXED = 0
    VARYING = 1


class ProfileType(Enum):
    """Profile type."""

    UNKNOWN = 0
    POLYGON = 1
    HORIZONTAL = 2
    VERTICAL = 4
    ORDER = 8


class DistanceSensorType(Enum):
    """Distance-sensor type."""

    NONE = 0
    SIMULATION = 1
    DLS_A_15 = 2
    DLS_A_15_FILTER = 3
    LSM215 = 4
    G500 = 5
    G600 = 6
    PSV700 = 7


class PreferredScanDirection(Enum):
    """Preferred scan direction."""

    NONE = 0
    X = 1
    Y = 2
    AUTO = 3
    INDEX = 4


class ScanHeadVideoType(IntFlag):
    """Scanning-head video capability/type flags."""

    NONE = 1
    REMOTE = 2
    COLOR = 4
    PAL = 8
    NTSC = 16
    DIGITAL = 32
    HD = 64


class ScanHeadCaps(IntFlag):
    """Scanning-head capability flags."""

    NONE = 0
    CONTROL = 1
    VIDEO = 2
    HAND_SET = 4
    PAN_TILT = 8
    LASER_FOCUS = 16
    LASER_AUTO_FOCUS = 32
    DISTANCE_SENSOR = 64
    MICROSCOPIC = 128
    DISPLAY = 256
    LASER_CONTROL = 512
    SCANNER_CONTROL = 1024
    HEADS_3_IN_ONE = 2048
    MULTIPLE_LENSES = 4096
    GREEN_LASER = 8192
    SPECKLE_TRACKING = 16384


class GraphicFormatType(Enum):
    """Graphics export format."""

    BITMAP = 0
    TIFF = 1
    PNG = 2
    TARGA = 3
    METAFILE = 4
    PHOTOSHOP30 = 5
    PCX = 6
    JPEG = 7
    POSTSCRIPT = 8
    WORDPERFECT = 9
    SUN = 10
    MACINTOSH_PICT = 11
    GIF = 12


class LensCalibrationCaps(IntFlag):
    """Lens-calibration capability flags."""

    INTERFERENCE = 1


class ChannelCaps(IntFlag):
    """Signal channel capability flags."""

    NONE = 0
    SCALAR = 1
    VECTOR = 2
    USER = 4
    X = 8
    Y = 16
    Z = 32
    D3_1D = 64
    RESPONSE = 128


class DomainType(Enum):
    """Signal domain type."""

    NOT_AVAIL = 0
    TIME = 1
    ANGLE_SYNC = 2
    SPECTRUM = 3
    THIRD_OCTAVE = 4
    ORDER_SPECTRUM = 5
    RMS = 6


class DisplayType(Enum):
    """Display type for a signal."""

    NOT_AVAIL = 0
    MAG = 1
    MAG_PHASE = 2
    MAG_DB = 3
    MAG_DB_PHASE = 4
    MAG_DB_A = 5
    PHASE = 6
    REAL = 7
    IMAG = 8
    REAL_IMAG = 9
    NYQUIST = 10
    INST_VAL = 11
    SAMPLES = 12
    IQ_FITTED = 13
    MAG_PHASE_RAD = 14
    MAG_DB_PHASE_RAD = 15


class DataType(Enum):
    """Signal data type."""

    UNKNOWN = 0
    POINT = 1
    AVERAGE = 2
    BAND = 3


class DOFDirection(Enum):
    """Degree-of-freedom direction."""

    SCALAR = 0
    PLUS_X_TRANSLATION = 1
    PLUS_Y_TRANSLATION = 2
    PLUS_Z_TRANSLATION = 3
    PLUS_X_ROTATION = 4
    PLUS_Y_ROTATION = 5
    PLUS_Z_ROTATION = 6
    VECTOR = 7
    MINUS_X_TRANSLATION = -1
    MINUS_Y_TRANSLATION = -2
    MINUS_Z_TRANSLATION = -3
    MINUS_X_ROTATION = -4
    MINUS_Y_ROTATION = -5
    MINUS_Z_ROTATION = -6


class FunctionType(Enum):
    """Signal function type."""

    UNKNOWN = 0
    TIME_RESPONSE = 1
    AUTO_SPECTRUM = 2
    CROSS_SPECTRUM = 3
    FREQUENCY_RESPONSE_FUNCTION = 4
    TRANSMISSIBILITY = 5
    COHERENCE = 6
    AUTO_CORRELATION = 7
    CROSS_CORRELATION = 8
    POWER_SPECTRAL_DENSITY = 9
    ENERGY_SPECTRAL_DENSITY = 10
    PROBABILITY_DENSITY_FUNCTION = 11
    SPECTRUM = 12
    CUMULATIVE_FREQUENCY_DISTRIBUTION = 13
    PEAKS_VALLEY = 14
    STRESS_CYCLES = 15
    STRAIN_CYCLES = 16
    ORBIT = 17
    MODE_INDICATOR_FUNCTION = 18
    FORCE_PATTERN = 19
    PARTIAL_POWER = 20
    PARTIAL_COHERENCE = 21
    EIGENVALUE = 22
    EIGENVECTOR = 23
    SHOCK_RESPONSE_SPECTRUM = 24
    FINITE_IMPULSE_RESPONSE_FILTER = 25
    MULTIPLE_COHERENCE = 26
    ORDER_FUNCTION = 27
    IMPULSE_RESPONSE = 100
    INTENSITY = 101
    PI_INDEX = 102
    SOUND_PRESSURE_LEVEL = 103
    SOUND_POWER = 104
    REAL_VALUED_DATA = 105
    COMPLEX_VALUED_DATA = 106
    PARTICLE_VELOCITY = 107
    TEMPERATURE = 108
    FREQUENCY_RESPONSE_FUNCTION_H1 = 300
    FREQUENCY_RESPONSE_FUNCTION_H2 = 301
    FREQUENCY_RESPONSE_FUNCTION_PCA_H1 = 302
    PCA_PRINCIPAL_INPUTS = 303
    PCA_VIRTUAL_COHERENCES = 304
    VDD_IQ = 305
    BAND = 306


def _flags_list(value: int, flag_enum: type[_FlagT], *, include_none: bool = False) -> list[_FlagT]:
    """Return the list of set flags for an ``IntFlag`` enum value."""
    flags = flag_enum(int(value))
    out: list[_FlagT] = []
    zero = flag_enum(0)
    for flag in flag_enum:
        if flag == zero:
            if include_none:
                out.append(flag)
            continue
        if flags & flag:
            out.append(flag)
    return out


def _unwrap_com(obj: Any) -> Any:
    """Return the raw COM object if ``obj`` is one of this module's wrappers."""
    return obj._com if hasattr(obj, "_com") else obj


def _build_linear_abscissa(x_axis: Any) -> np.ndarray:
    """Build a linearly spaced x-axis array from a COM x-axis descriptor."""
    count = int(x_axis.max_count)
    if count <= 0:
        return np.array([], dtype=float)
    if count == 1:
        return np.array([float(x_axis.min)], dtype=float)
    step = (x_axis.max - x_axis.min) / (count - 1)
    return np.arange(count, dtype=float) * step + x_axis.min


def _find_display_by_name(signal: Any, display_name: str) -> Any:
    """Return the display with the requested name or raise ``ValueError``."""
    for display in signal.displays:
        if display.name == display_name:
            return display
    raise ValueError(f"No display named {display_name!r} found for signal " f"{signal.name!r}.")


class AverageProperties:
    """Wrapper around COM average settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def count(self) -> int:
        """Number of averages in the measurement."""
        return int(self._com.Count)

    @property
    def type(self) -> AverageType:
        """Averaging type used for the measurement."""
        return AverageType(self._com.Type)


class DigitalFilter:
    """Wrapper around COM digital filter settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def type(self) -> DigitalFilterType:
        """Filter type."""
        return DigitalFilterType(self._com.Type)

    @property
    def quality(self) -> DigitalFilterQuality:
        """Filter quality setting."""
        return DigitalFilterQuality(self._com.Quality)

    @property
    def cutoff_freq(self) -> float | tuple[float, float]:
        """Cutoff frequency or frequencies for the configured filter.

        Returns
        -------
        float | tuple[float, float]
            A single cutoff frequency for low-pass and high-pass filters, or a
            pair of cutoff frequencies for band-pass and notch filters.

        Raises
        ------
        ValueError
            If the filter type does not define a cutoff frequency.
        """
        filter_type = self.type
        if filter_type in (DigitalFilterType.LOWPASS, DigitalFilterType.HIGHPASS):
            return float(self._com.CutoffFreq)
        if filter_type in (DigitalFilterType.BANDPASS, DigitalFilterType.NOTCH):
            return (float(self._com.CutoffFreq1), float(self._com.CutoffFreq2))
        raise ValueError("Filter has no cutoff frequency defined.")


class ChannelsProperties:
    """Wrapper around COM acquisition channel properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def active(self) -> bool:
        """Whether the channel is active."""
        return bool(self._com.Active)

    @property
    def calibration(self) -> float:
        """Calibration factor in SI units."""
        return float(self._com.Calibration)

    @property
    def differential_input(self) -> bool:
        """Whether the channel expects differential input."""
        return bool(self._com.DifferentialInput)

    @property
    def digital_filter(self) -> DigitalFilter:
        """Digital filter applied to the input signal."""
        return DigitalFilter(self._com.DigitalFilter)

    @property
    def direction(self) -> Direction:
        """Direction of vibration for this channel."""
        return Direction(self._com.DirectionOfVibration)

    @property
    def icp_input(self) -> bool:
        """Whether the channel is configured for an ICP device."""
        return bool(self._com.ICPInput)

    @property
    def coupling(self) -> Coupling:
        """Input coupling for the channel."""
        return Coupling(self._com.InputCoupling)

    @property
    def impedance(self) -> float:
        """Input impedance in ohms."""
        return float(self._com.InputImpedance)

    @property
    def range(self) -> float:
        """Input range in volts."""
        return float(self._com.InputRange)

    @property
    def int_diff_physical_quantity(self) -> PhysicalQuantity:
        """Quantity produced by integration/differentiation filtering."""
        return PhysicalQuantity(self._com.IntDiffPhysicalQuantity)

    @property
    def name(self) -> str:
        """Channel descriptor."""
        return self._com.Name

    @property
    def quantity(self) -> PhysicalQuantity:
        """Physical quantity of the channel."""
        return PhysicalQuantity(self._com.Quantity)

    @property
    def reference(self) -> bool:
        """Whether the channel is marked as a reference."""
        return bool(self._com.Reference)

    @property
    def node_id(self) -> int:
        """One-based scan-point index associated with the channel."""
        return int(self._com.ReferencePointIndex)

    @property
    def signal_enhancement(self) -> bool:
        """Whether signal enhancement is active for the channel."""
        return bool(self._com.SEActive)

    @property
    def short_name(self) -> str:
        """Abbreviated channel descriptor."""
        return self._com.ShortName

    @property
    def signal_delay(self) -> float:
        """Signal delay in seconds."""
        return float(self._com.SignalDelay)

    @property
    def type(self) -> ChannelType:
        """Signal type of the channel."""
        return ChannelType(self._com.Type)

    @property
    def unit(self) -> str:
        """Engineering unit of the channel quantity."""
        return self._com.Unit

    @property
    def vibrometer_index(self) -> int:
        """Index of the vibrometer connected to this channel, or 0 if none."""
        return int(self._com.VibrometerIndex)

    @property
    def window_function(self) -> WindowFunction:
        """Window function used for the channel."""
        return WindowFunction(self._com.WindowFunction)

    @property
    def window_function_parameters(self) -> Any:
        """Parameters of the configured window function."""
        return self._com.WindowFunctionParams

    @property
    def window_function_rms_correction(self) -> float:
        """RMS correction factor for the configured window function."""
        return float(self._com.WindowFunctionRMSCorrection)


class FastScanProperties:
    """Wrapper around COM FastScan acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def bandwidth(self) -> float:
        """FastScan bandwidth in hertz."""
        return float(self._com.Bandwidth)

    @property
    def frequency(self) -> float:
        """FastScan frequency in hertz."""
        return float(self._com.Frequency)

    @property
    def sample_frequency(self) -> float:
        """Sampling frequency in hertz."""
        return float(self._com.SampleFrequency)

    @property
    def samples(self) -> int:
        """Number of samples in the FastScan acquisition."""
        return int(self._com.Samples)

    @property
    def sample_time(self) -> float:
        """Sample time in seconds."""
        return float(self._com.SampleTime)


class FFTProperties:
    """Wrapper around COM FFT acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def bandwidth(self) -> float:
        """FFT bandwidth in hertz."""
        return float(self._com.Bandwidth)

    @property
    def end_frequency(self) -> float:
        """End frequency in hertz."""
        return float(self._com.EndFrequency)

    @property
    def lines(self) -> int:
        """Number of FFT lines."""
        return int(self._com.Lines)

    @property
    def overlap(self) -> float:
        """Overlap in percent."""
        return float(self._com.Overlap)

    @property
    def sample_frequency(self) -> float:
        """Sampling frequency in hertz."""
        return float(self._com.SampleFrequency)

    @property
    def sample_resolution(self) -> float:
        """Frequency resolution in hertz."""
        return float(self._com.SampleResolution)

    @property
    def samples(self) -> int:
        """Number of samples used for the FFT."""
        return int(self._com.Samples)

    @property
    def sample_time(self) -> float:
        """Sample time in seconds."""
        return float(self._com.SampleTime)

    @property
    def start_frequency(self) -> float:
        """Start frequency in hertz."""
        return float(self._com.StartFrequency)


class FrontEndProperties:
    """Wrapper around COM front-end acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def generator_amplifier_enabled(self) -> bool:
        """Whether the generator amplifier is enabled."""
        return bool(self._com.GeneratorAmplifierEnabled)

    @property
    def generator_amplifier_factor(self) -> float:
        """Amplification factor of the generator amplifier."""
        return float(self._com.GeneratorAmplifierFactor)


class GeneralProperties:
    """Wrapper around COM general acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def auto_remeasure(self) -> bool:
        """Whether auto remeasure is enabled."""
        return bool(self._com.AutoRemeasure)

    @property
    def auto_remeasure_auto_range(self) -> bool:
        """Whether vibrometer range is automatically adjusted on overrange."""
        return bool(self._com.AutoRemeasureAutoRange)

    @property
    def bandwidth_extension(self) -> BandwidthExtension:
        """Bandwidth extension setting for the UHF vibrometer."""
        return BandwidthExtension(self._com.BandWidthExtension)

    @property
    def principal_component_analysis(self) -> bool:
        """Whether principal component analysis (MIMO) is enabled."""
        return bool(self._com.PrincipalComponentAnalysis)

    @property
    def real_sample_frequency(self) -> float:
        """Raw sampling frequency before digital decimation."""
        return float(self._com.RealSampleFrequency)


class GeneratorsProperties:
    """Wrapper around COM signal-generator properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def active(self) -> bool:
        """Whether the generator is active."""
        return bool(self._com.Active)

    @property
    def amplitude(self) -> float:
        """Generator amplitude in volts."""
        return float(self._com.Amplitude)

    @property
    def amplitude_correction_data(self) -> Any:
        """Amplitude-correction data."""
        return self._com.AmplitudeCorrectionData

    @property
    def multiple_channels(self) -> bool:
        """Whether the generator uses multiple output channels."""
        return bool(self._com.MultipleChannels)

    @property
    def offset(self) -> float:
        """Generator DC offset in volts."""
        return float(self._com.Offset)

    @property
    def steady_state_time(self) -> float:
        """Waiting time for steady-state conditions in seconds."""
        return float(self._com.SteadyStateTime)

    @property
    def supports_multiple_channels(self) -> bool:
        """Whether the generator supports multiple output channels."""
        return bool(self._com.SupportsMultipleChannels)

    @property
    def use_amplitude_correction(self) -> bool:
        """Whether amplitude-correction data is used."""
        return bool(self._com.UseAmplitudeCorrection)

    @property
    def waveform(self) -> "Waveform":
        """Configured waveform."""
        return make_waveform(self._com.Waveform)


class Waveform:
    """Base wrapper for generator waveform settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def name(self) -> str:
        """Waveform name."""
        return self._com.Name

    @property
    def type(self) -> WaveformType:
        """Waveform type."""
        return WaveformType(self._com.Type)


class BurstChirpWaveform(Waveform):
    """Burst-chirp waveform settings."""

    @property
    def burst_length(self) -> float:
        """Burst length in percent."""
        return float(self._com.BurstLength)

    @property
    def burst_length_ex(self) -> float:
        """Extended burst length in percent."""
        return float(self._com.BurstLengthEx)

    @property
    def burst_start(self) -> float:
        """Burst start in percent."""
        return float(self._com.BurstStart)

    @property
    def burst_start_ex(self) -> float:
        """Extended burst start in percent."""
        return float(self._com.BurstStartEx)

    @property
    def end_frequency(self) -> float:
        """End frequency in hertz."""
        return float(self._com.EndFrequency)

    @property
    def start_frequency(self) -> float:
        """Start frequency in hertz."""
        return float(self._com.StartFrequency)


class BurstRandomWaveform(Waveform):
    """Burst-random waveform settings."""

    @property
    def burst_length(self) -> float:
        """Burst length in percent."""
        return float(self._com.BurstLength)

    @property
    def burst_length_ex(self) -> float:
        """Extended burst length in percent."""
        return float(self._com.BurstLengthEx)

    @property
    def burst_start(self) -> float:
        """Burst start in percent."""
        return float(self._com.BurstStart)

    @property
    def burst_start_ex(self) -> float:
        """Extended burst start in percent."""
        return float(self._com.BurstStartEx)


class MultiCarrierCWWaveform(Waveform):
    """Multi-carrier continuous-wave waveform settings."""

    @property
    def end_frequency(self) -> float:
        """End frequency in hertz."""
        return float(self._com.EndFrequency)

    @property
    def start_frequency(self) -> float:
        """Start frequency in hertz."""
        return float(self._com.StartFrequency)


class PeriodicChirpWaveform(Waveform):
    """Periodic-chirp waveform."""


class PseudoRandomWaveform(Waveform):
    """Pseudo-random waveform."""


class RampWaveform(Waveform):
    """Ramp waveform settings."""

    @property
    def frequency(self) -> float:
        """Waveform frequency in hertz."""
        return float(self._com.Frequency)


class SineWaveform(Waveform):
    """Sine waveform settings."""

    @property
    def frequency(self) -> float:
        """Waveform frequency in hertz."""
        return float(self._com.Frequency)


class SquareWaveform(Waveform):
    """Square waveform settings."""

    @property
    def frequency(self) -> float:
        """Waveform frequency in hertz."""
        return float(self._com.Frequency)


class SweepWaveform(Waveform):
    """Sweep waveform settings."""

    @property
    def end_frequency(self) -> float:
        """End frequency of the sweep in hertz."""
        return float(self._com.EndFrequency)

    @property
    def start_frequency(self) -> float:
        """Start frequency of the sweep in hertz."""
        return float(self._com.StartFrequency)

    @property
    def sweep_time(self) -> float:
        """Sweep duration in seconds."""
        return float(self._com.SweepTime)


class TriangleWaveform(Waveform):
    """Triangle waveform settings."""

    @property
    def frequency(self) -> float:
        """Waveform frequency in hertz."""
        return float(self._com.Frequency)


class WhiteNoiseWaveform(Waveform):
    """White-noise waveform."""


class UserDefinedWaveform(Waveform):
    """User-defined waveform settings."""

    @property
    def frequency(self) -> float:
        """Waveform frequency in hertz."""
        return float(self._com.Frequency)

    @property
    def waveform_data(self) -> Any:
        """Waveform data for the user-defined waveform."""
        return self._com.WaveformData


def make_waveform(waveform_com_obj: Any) -> Waveform:
    """Wrap a COM waveform object in the appropriate Python subclass."""
    wf_type = WaveformType(waveform_com_obj.Type)

    cls_map: dict[WaveformType, type[Waveform]] = {
        WaveformType.SINE: SineWaveform,
        WaveformType.SQUARE: SquareWaveform,
        WaveformType.TRIANGLE: TriangleWaveform,
        WaveformType.RAMP: RampWaveform,
        WaveformType.SWEEP: SweepWaveform,
        WaveformType.BURST_CHIRP: BurstChirpWaveform,
        WaveformType.BURST_RANDOM: BurstRandomWaveform,
        WaveformType.PERIODIC_CHIRP: PeriodicChirpWaveform,
        WaveformType.PSEUDORANDOM: PseudoRandomWaveform,
        WaveformType.WHITE_NOISE: WhiteNoiseWaveform,
        WaveformType.USER_DEFINED: UserDefinedWaveform,
        WaveformType.MULTI_CARRIER_CW: MultiCarrierCWWaveform,
    }

    cls = cls_map.get(wf_type, Waveform)
    return cls(waveform_com_obj)


class MultiFrameProperties:
    """Wrapper around COM multi-frame acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def average_type(self) -> AverageType:
        """Average type used to average frames."""
        return AverageType(self._com.AverageType)

    @property
    def frame_count(self) -> int:
        """Number of frames."""
        return int(self._com.FrameCount)

    @property
    def mode(self) -> MultiFrameMode:
        """Multi-frame mode."""
        return MultiFrameMode(self._com.Mode)

    @property
    def pulses_per_cycle(self) -> int:
        """Pulses per engine cycle used in automatic frame mode."""
        return int(self._com.PulsesPerCycle)


class SignalEnhancementProperties:
    """Wrapper around COM signal-enhancement properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def mode(self) -> SignalEnhancementMode:
        """Signal-enhancement mode."""
        return SignalEnhancementMode(self._com.Mode)

    @property
    def speckle_tracking(self) -> bool:
        """Whether speckle tracking is active."""
        return bool(self._com.SpeckleTracking)


class TimeProperties:
    """Wrapper around COM time-acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def sample_frequency(self) -> float:
        """Sample frequency in hertz."""
        return float(self._com.SampleFrequency)

    @property
    def sample_resolution(self) -> float:
        """Time resolution in seconds."""
        return float(self._com.SampleResolution)

    @property
    def samples(self) -> int:
        """Number of samples."""
        return int(self._com.Samples)

    @property
    def sample_time(self) -> float:
        """Sample time in seconds."""
        return float(self._com.SampleTime)


class TriggerProperties:
    """Wrapper around COM trigger properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def analog_source_channel(self) -> int:
        """Source channel for analog triggering."""
        return int(self._com.AnalogSourceChannel)

    @property
    def edge(self) -> TriggerEdge:
        """Trigger edge."""
        return TriggerEdge(self._com.Edge)

    @property
    def level(self) -> float:
        """Trigger level in percent."""
        return float(self._com.Level)

    @property
    def phase_from_ref(self) -> bool:
        """Whether phase is determined from the first reference FRF."""
        return bool(self._com.PhaseFromRef)

    @property
    def pre_trigger(self) -> float:
        """Pre-trigger value in percent."""
        return float(self._com.PreTrigger)

    @property
    def pre_trigger_max(self) -> float:
        """Maximum pre-trigger value in percent."""
        return float(self._com.PreTriggerMax)

    @property
    def pre_trigger_min(self) -> float:
        """Minimum pre-trigger value in percent."""
        return float(self._com.PreTriggerMin)

    @property
    def pre_trigger_possible(self) -> bool:
        """Whether pre-trigger is possible for the current settings."""
        return bool(self._com.PreTriggerPossible)

    @property
    def source(self) -> TriggerSource:
        """Trigger source."""
        return TriggerSource(self._com.Source)


class ControllerInfo:
    """Wrapper around COM controller-information settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def caps_value(self) -> int:
        """Controller capabilities as a raw integer bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> ControllerCaps:
        """Controller capabilities as an ``IntFlag``."""
        return decode_controller_caps(self.caps_value)

    @property
    def caps_set(self) -> list[ControllerCaps]:
        """Individual controller capability flags that are set."""
        return controller_caps_list(self.caps_value)

    def has_cap(self, cap: ControllerCaps) -> bool:
        """Return whether the controller supports the given capability flag(s)."""
        return bool(self.caps & cap)

    @property
    def device_name(self) -> str:
        """General controller device name."""
        return self._com.DeviceName

    @property
    def firmware_version(self) -> str:
        """Controller firmware version."""
        return self._com.FirmwareVersion

    @property
    def name(self) -> str:
        """Controller type/name."""
        return self._com.Name

    @property
    def serial_number(self) -> str:
        """Controller serial number."""
        return self._com.SerialNumber

    @property
    def unique_id(self) -> str:
        """Controller unique identifier."""
        return self._com.UniqueID


class VideoCameraSettings:
    """Wrapper around COM video-camera settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def automatic_gain_control(self) -> bool:
        """Whether automatic gain control is active."""
        return bool(self._com.AutomaticGainControl)

    @property
    def gamma(self) -> float:
        """Gamma value."""
        return float(self._com.Gamma)

    @property
    def manual_gain(self) -> float:
        """Manual gain value."""
        return float(self._com.ManualGain)

    @property
    def mirror(self) -> bool:
        """Mirror mode state."""
        return bool(self._com.Mirror)

    @property
    def shutter_speed(self) -> float:
        """Shutter speed."""
        return float(self._com.ShutterSpeed)


class SensorHeadInfo:
    """Wrapper around COM sensor-head information."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def beam_gap(self) -> Any:
        """Beam gap, including units when provided by the COM interface."""
        return self._com.BeamGap

    @property
    def bragg_cell_frequency(self) -> float:
        """Bragg-cell frequency."""
        return float(self._com.BraggCellFrequency)

    @property
    def caps_value(self) -> int:
        """Sensor-head capabilities as a raw integer bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> SensorHeadCaps:
        """Sensor-head capabilities as an ``IntFlag``."""
        return SensorHeadCaps(self.caps_value)

    @property
    def caps_set(self) -> list[SensorHeadCaps]:
        """Individual sensor-head capability flags that are set."""
        return _flags_list(self.caps_value, SensorHeadCaps)

    @property
    def caps2_value(self) -> int:
        """Extended sensor-head capabilities as a raw integer bitmask."""
        return int(self._com.Caps2)

    @property
    def caps2(self) -> SensorHeadCaps2:
        """Extended sensor-head capabilities as an ``IntFlag``."""
        return SensorHeadCaps2(self.caps2_value)

    @property
    def caps2_set(self) -> list[SensorHeadCaps2]:
        """Individual extended sensor-head capability flags that are set."""
        return _flags_list(self.caps2_value, SensorHeadCaps2)

    def has_cap(self, cap: SensorHeadCaps) -> bool:
        """Return whether the sensor head supports the given capability flag(s)."""
        return bool(self.caps & cap)

    def has_cap2(self, cap: SensorHeadCaps2) -> bool:
        """Return whether the sensor head supports the given extended capability flag(s)."""
        return bool(self.caps2 & cap)

    @property
    def coherence_optimizer_firmware_version(self) -> str:
        """Firmware version of the coherence optimizer."""
        return self._com.CoherenceOptimizerFirmwareVersion

    @property
    def connected(self) -> bool:
        """Whether the sensor head is connected."""
        return bool(self._com.Connected)

    @property
    def dimmer_max(self) -> float:
        """Maximum laser-beam transmission."""
        return float(self._com.DimmerMax)

    @property
    def dimmer_min(self) -> float:
        """Minimum laser-beam transmission."""
        return float(self._com.DimmerMin)

    @property
    def firmware_version(self) -> str:
        """Sensor-head firmware version."""
        return self._com.FirmwareVersion

    @property
    def focus_far_limit(self) -> float:
        """Farthest focus position."""
        return float(self._com.FocusFarLimit)

    @property
    def focus_near_limit(self) -> float:
        """Nearest focus position."""
        return float(self._com.FocusNearLimit)

    @property
    def illumination_intensity_max(self) -> float:
        """Maximum illumination intensity."""
        return float(self._com.IlluminationIntensityMax)

    @property
    def illumination_intensity_min(self) -> float:
        """Minimum illumination intensity."""
        return float(self._com.IlluminationIntensityMin)

    @property
    def laser_delay(self) -> float:
        """Laser delay in seconds."""
        return float(self._com.LaserDelay)

    @property
    def laser_firmware_version(self) -> str:
        """Laser firmware version string."""
        return self._com.LaserFirmwareVersion

    @property
    def laser_power(self) -> float:
        """Laser power in milliwatts."""
        return float(self._com.LaserPower)

    @property
    def laser_wavelength(self) -> float:
        """Laser wavelength in nanometers."""
        return float(self._com.LaserWavelength)

    @property
    def lower_objective_threshold_temperature(self) -> float:
        """Lower objective-threshold temperature."""
        return float(self._com.LowerObjectiveThresholdTemperature)

    @property
    def max_rgb_led_color_index(self) -> int:
        """Maximum RGB LED color index."""
        return int(self._com.MaxRGBLedColorIndex)

    @property
    def max_signal_balance(self) -> SignalBalanceState:
        """Upper end of the signal-balance range."""
        return SignalBalanceState(self._com.MaxSignalBalance)

    @property
    def min_rgb_led_color_index(self) -> int:
        """Minimum RGB LED color index."""
        return int(self._com.MinRGBLedColorIndex)

    @property
    def min_signal_balance(self) -> SignalBalanceState:
        """Lower end of the signal-balance range."""
        return SignalBalanceState(self._com.MinSignalBalance)

    @property
    def name(self) -> str:
        """Sensor-head name."""
        return self._com.Name

    @property
    def serial_number(self) -> str:
        """Sensor-head serial number."""
        return self._com.SerialNumber

    @property
    def sonde_name(self) -> str:
        """Sonde name."""
        return self._com.SondeName

    @property
    def standoff_dist(self) -> Any:
        """Stand-off distance, including units when provided by the COM interface."""
        return self._com.StandoffDist

    @property
    def system_laser_power(self) -> float:
        """System laser power in milliwatts."""
        return float(self._com.SystemLaserPower)

    @property
    def unique_id(self) -> str:
        """Unique sensor-head identifier."""
        return self._com.UniqueID

    @property
    def upper_objective_threshold_temperature(self) -> float:
        """Upper objective-threshold temperature."""
        return float(self._com.UpperObjectiveThresholdTemperature)


class SensorHeadSettings:
    """Wrapper around COM sensor-head settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def dimmer(self) -> float:
        """Dimmer value."""
        return float(self._com.Dimmer)

    @property
    def illumination_intensity(self) -> float:
        """Illumination intensity."""
        return float(self._com.IlluminationIntensity)

    @property
    def illumination_state(self) -> Any:
        """Illumination state."""
        return self._com.IlluminationState

    @property
    def key(self) -> Any:
        """Sensor-head identifier key."""
        return self._com.Key

    @property
    def name(self) -> str:
        """Sensor-head name."""
        return self._com.Name

    @property
    def sensor_head_info(self) -> SensorHeadInfo:
        """Information object for the sensor head."""
        return SensorHeadInfo(self._com.SensorHeadInfo)

    @property
    def video_camera_settings(self) -> VideoCameraSettings:
        """Video-camera settings for the sensor head.

        Raises
        ------
        ValueError
            If no video-camera settings are available.
        """
        if self._com.VideoCameraSettings is None:
            raise ValueError(f"No video camera settings found for {self!r}")
        return VideoCameraSettings(self._com.VideoCameraSettings)


class FilterSettings:
    """Wrapper around COM controller-filter settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def key(self) -> FilterType:
        """Filter type key."""
        return FilterType(self._com.Key)

    @property
    def name(self) -> str:
        """Filter name."""
        return self._com.Name

    @property
    def range(self) -> Any:
        """Filter range setting."""
        return self._com.Range


class DecoderInfo:
    """Wrapper around COM decoder information."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def firmware_version(self) -> str:
        """Decoder firmware version."""
        return self._com.FirmwareVersion

    @property
    def name(self) -> str:
        """Decoder name."""
        return self._com.Name


class QuantitySettings:
    """Wrapper around COM quantity settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def bandwidth_range(self) -> str:
        """Active decoder bandwidth range name."""
        return self._com.BandwidthRange

    @property
    def decoder_info(self) -> DecoderInfo:
        """Decoder information for this quantity."""
        return DecoderInfo(self._com.DecoderInfo)

    @property
    def key(self) -> QuantityType:
        """Quantity-type key."""
        return QuantityType(self._com.Key)

    @property
    def max_velocity_range(self) -> float:
        """Maximum velocity range."""
        return float(self._com.MaxVelocityRange)

    @property
    def name(self) -> str:
        """Quantity name."""
        return self._com.Name

    @property
    def output_active(self) -> bool:
        """Whether output of the quantity is active."""
        return bool(self._com.OutputActive)

    @property
    def overrun(self) -> Any:
        """Overrun mode/state."""
        return self._com.Overrun

    @property
    def range(self) -> str:
        """Active decoder range name."""
        return self._com.Range


class ControllerSettings:
    """Wrapper around COM vibrometer-controller settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def controller_info(self) -> ControllerInfo:
        """Controller information."""
        return ControllerInfo(self._com.ControllerInfo)

    @property
    def decoder_clear_mode(self) -> Any:
        """Decoder clear mode."""
        return self._com.DecoderClearMode

    @property
    def filter_settings(self) -> list[FilterSettings]:
        """Filter settings."""
        return [FilterSettings(x) for x in self._com.FilterSettingsCollection]

    @property
    def quantity_settings(self) -> list[QuantitySettings]:
        """Quantity settings."""
        return [QuantitySettings(x) for x in self._com.QuantitySettingsCollection]

    @property
    def sensor_head_settings(self) -> list[SensorHeadSettings]:
        """Sensor-head settings."""
        return [SensorHeadSettings(x) for x in self._com.SensorHeadSettingsCollection]


class VibrometerProperties:
    """Wrapper around COM vibrometer acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def controller_settings(self) -> ControllerSettings:
        """Vibrometer controller settings."""
        return ControllerSettings(self._com.VibControllerSettings)


class ZoomFFTProperties:
    """Wrapper around COM zoom-FFT acquisition properties."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def bandwidth(self) -> float:
        """Zoom-FFT bandwidth in hertz."""
        return float(self._com.Bandwidth)

    @property
    def center_frequency(self) -> float:
        """Zoom-FFT center frequency in hertz."""
        return float(self._com.CenterFrequency)

    @property
    def lines(self) -> int:
        """Number of Zoom-FFT lines."""
        return int(self._com.Lines)

    @property
    def overlap(self) -> float:
        """Zoom-FFT overlap in percent."""
        return float(self._com.Overlap)

    @property
    def sample_frequency(self) -> float:
        """Sample frequency in hertz."""
        return float(self._com.SampleFrequency)

    @property
    def sample_resolution(self) -> float:
        """Frequency resolution in hertz."""
        return float(self._com.SampleResolution)

    @property
    def samples(self) -> int:
        """Number of samples used for the Zoom-FFT."""
        return int(self._com.Samples)

    @property
    def sample_time(self) -> float:
        """Sample time in seconds."""
        return float(self._com.SampleTime)


# Alignments
# Alignments


class AlignmentPoint2D:
    """Wrapper around a COM 2D alignment point."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def scanner_x(self) -> float:
        """Horizontal scanner value of the 2D alignment point."""
        return float(self._com.ScannerX)

    @property
    def scanner_y(self) -> float:
        """Vertical scanner value of the 2D alignment point."""
        return float(self._com.ScannerY)

    @property
    def video_x(self) -> float:
        """X video coordinate of the 2D alignment point."""
        return float(self._com.VideoX)

    @property
    def video_y(self) -> float:
        """Y video coordinate of the 2D alignment point."""
        return float(self._com.VideoY)


class Alignment2D:
    """Wrapper around COM 2D alignment data and operations."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def align_2d_points(self) -> list[AlignmentPoint2D]:
        """2D alignment points."""
        return [AlignmentPoint2D(p) for p in self._com.Align2DPoints]

    @property
    def scan_head_distance(self) -> float:
        """Distance from the scan head front panel to the object, in meters."""
        return float(self._com.ScanHeadDistance)

    @property
    def scan_head_type(self) -> ScanHeadType:
        """Type of scanning head used for the alignment."""
        return ScanHeadType(self._com.ScanHeadType)

    @property
    def scanner_quantity(self) -> Any:
        """Physical quantity represented by the scanner coordinates."""
        return self._com.ScannerQuantity

    @property
    def valid(self) -> bool:
        """Whether the 2D alignment is valid."""
        return bool(self._com.Valid)

    def calculate(self) -> None:
        """Calculate a new 2D alignment."""
        self._com.Calculate()

    def get_cosine_correction_factor(
        self,
        scanner_x: float,
        scanner_y: float,
        vib_correct_x: bool,
        vib_correct_y: bool,
    ) -> float:
        """Return the cosine-correction factor for the given scanner coordinates."""
        return float(
            self._com.GetCosineCorrectionFactor(scanner_x, scanner_y, vib_correct_x, vib_correct_y)
        )

    def invalidate(self) -> None:
        """Invalidate the current 2D alignment and reset related properties."""
        self._com.Invalidate()

    def scanner_to_video(self, scanner_x: float, scanner_y: float) -> Any:
        """Convert scanner coordinates to video coordinates."""
        return self._com.ScannerToVideo(scanner_x, scanner_y)

    def video_to_scanner(self, video_x: float, video_y: float) -> Any:
        """Convert video coordinates to scanner coordinates."""
        return self._com.VideoToScanner(video_x, video_y)


class AlignmentPoint3D:
    """Wrapper around a COM 3D alignment point."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def caps_value(self) -> int:
        """Capabilities of the 3D alignment point as a raw bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> AlignmentPoint3DCaps:
        """Capabilities of the 3D alignment point as an ``IntFlag``."""
        return AlignmentPoint3DCaps(self.caps_value)

    @property
    def distance(self) -> float:
        """Distance associated with the alignment point."""
        return float(self._com.Distance)

    @property
    def label(self) -> str:
        """Alignment-point label."""
        return self._com.Label

    @property
    def point_type(self) -> AlignmentPoint3DType:
        """Alignment-point type."""
        return AlignmentPoint3DType(self._com.PointType)

    @property
    def quality(self) -> float:
        """Alignment quality in meters."""
        return float(self._com.Quality)

    @property
    def scanner_x(self) -> float:
        """Horizontal scanner value of the 3D alignment point."""
        return float(self._com.ScannerX)

    @property
    def scanner_y(self) -> float:
        """Vertical scanner value of the 3D alignment point."""
        return float(self._com.ScannerY)

    @property
    def video_x(self) -> float:
        """X video coordinate of the 3D alignment point."""
        return float(self._com.VideoX)

    @property
    def video_y(self) -> float:
        """Y video coordinate of the 3D alignment point."""
        return float(self._com.VideoY)

    @property
    def x(self) -> float:
        """X coordinate of the 3D alignment point."""
        return float(self._com.X)

    @property
    def y(self) -> float:
        """Y coordinate of the 3D alignment point."""
        return float(self._com.Y)

    @property
    def z(self) -> float:
        """Z coordinate of the 3D alignment point."""
        return float(self._com.Z)


class Alignment3D:
    """Wrapper around COM 3D alignment data and operations."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def align_3d_points(self) -> list[AlignmentPoint3D]:
        """3D alignment points."""
        return [AlignmentPoint3D(p) for p in self._com.Align3DPoints]

    @property
    def coord_definition_mode(self) -> CoordDefinitionMode:
        """Coordinate-definition mode for the 3D alignment."""
        return CoordDefinitionMode(self._com.CoordDefinitionMode)

    @property
    def current_quality(self) -> float:
        """Current 3D alignment quality in meters."""
        return float(self._com.CurrentQuality)

    @property
    def mirror(self) -> bool:
        """Whether the 3D alignment was performed through a mirror."""
        return bool(self._com.Mirror)

    @property
    def scan_head_type(self) -> ScanHeadType:
        """Scanning-head type."""
        return ScanHeadType(self._com.ScanHeadType)

    @property
    def target_quality(self) -> float:
        """Target 3D alignment quality in meters."""
        return float(self._com.TargetQuality)

    @property
    def valid(self) -> bool:
        """Whether the alignment is valid."""
        return bool(self._com.Valid)

    def calculate(self) -> None:
        """Calculate a new 3D alignment."""
        self._com.Calculate()

    def coord_3d_to_scanner(self, x: float, y: float, z: float) -> Any:
        """Convert 3D coordinates to scanner coordinates and distance."""
        return self._com.Coord3DToScanner(x, y, z)

    def get_beam_length(self, x: float, y: float, z: float) -> float:
        """Retrieve beam length from the objective to the specified point."""
        return float(self._com.GetBeamLength(x, y, z))

    def get_beam_origin(self, scanner_x: float = 0.0, scanner_y: float = 0.0) -> Any:
        """Retrieve the beam origin at the second scanner mirror."""
        return self._com.GetBeamOrigin(scanner_x, scanner_y)

    def get_scan_head_position(self) -> Any:
        """Retrieve the scan-head position at the first scanner mirror."""
        return self._com.GetScanHeadPosition()

    def invalidate(self) -> None:
        """Invalidate the current 3D alignment."""
        self._com.Invalidate()

    def scanner_to_coord_3d(self, scanner_x: float, scanner_y: float, distance: float) -> Any:
        """Convert scanner coordinates and distance to 3D coordinates."""
        return self._com.ScannerToCoord3D(scanner_x, scanner_y, distance)

    def scanner_to_vector_3d(self, scanner_x: float, scanner_y: float) -> Any:
        """Convert scanner coordinates to a unit-direction vector."""
        return self._com.ScannerToVector3D(scanner_x, scanner_y)


class AlignmentCamera:
    """Wrapper around COM camera-alignment data and operations."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def align_3d_points(self) -> list[AlignmentPoint3D]:
        """3D alignment points."""
        return [AlignmentPoint3D(p) for p in self._com.Align3DPoints]

    @property
    def aspect_ratio(self) -> float:
        """Video width-to-height aspect ratio."""
        return float(self._com.AspectRatio)

    @property
    def cam_pos_x(self) -> float:
        """X coordinate of the camera position."""
        return float(self._com.CamPosX)

    @property
    def cam_pos_y(self) -> float:
        """Y coordinate of the camera position."""
        return float(self._com.CamPosY)

    @property
    def cam_pos_z(self) -> float:
        """Z coordinate of the camera position."""
        return float(self._com.CamPosZ)

    @property
    def cam_vec_x(self) -> float:
        """X coordinate of the camera view direction."""
        return float(self._com.CamVecX)

    @property
    def cam_vec_y(self) -> float:
        """Y coordinate of the camera view direction."""
        return float(self._com.CamVecY)

    @property
    def cam_vec_z(self) -> float:
        """Z coordinate of the camera view direction."""
        return float(self._com.CamVecZ)

    @property
    def current_quality(self) -> float:
        """Current alignment quality."""
        return float(self._com.CurrentQuality)

    @property
    def mirror(self) -> bool:
        """Whether the object is viewed through a mirror."""
        return bool(self._com.Mirror)

    @property
    def target_quality(self) -> float:
        """Target alignment quality."""
        return float(self._com.TargetQuality)

    @property
    def valid(self) -> bool:
        """Whether the current alignment is valid."""
        return bool(self._com.Valid)

    @property
    def view_angle(self) -> float:
        """Camera aperture angle in degrees."""
        return float(self._com.ViewAngle)

    @property
    def view_up_x(self) -> float:
        """X coordinate of the camera view-up direction."""
        return float(self._com.ViewUpX)

    @property
    def view_up_y(self) -> float:
        """Y coordinate of the camera view-up direction."""
        return float(self._com.ViewUpY)

    @property
    def view_up_z(self) -> float:
        """Z coordinate of the camera view-up direction."""
        return float(self._com.ViewUpZ)

    def calculate(self, *args: Any) -> Any:
        """Calculate camera parameters from input parameters and alignment points."""
        return self._com.Calculate(*args)

    def choose_camera_position(self, *args: Any) -> Any:
        """Choose a camera position to view the specified 3D points."""
        return self._com.ChooseCameraPosition(*args)

    def coord_3d_to_video(self, x: float, y: float, z: float) -> Any:
        """Convert 3D object coordinates to video coordinates."""
        return self._com.Coord3DToVideo(x, y, z)

    def scanner_to_video(self, *args: Any) -> Any:
        """Convert scanner coordinates to video coordinates."""
        return self._com.ScannerToVideo(*args)

    def set_camera(self, *args: Any) -> Any:
        """Set camera parameters."""
        return self._com.SetCamera(*args)

    def video_to_coord_3d(self, *args: Any) -> Any:
        """Convert video coordinates to 3D object coordinates."""
        return self._com.VideoToCoord3D(*args)

    def video_to_scanner(self, *args: Any) -> Any:
        """Convert video coordinates to scanner coordinates."""
        return self._com.VideoToScanner(*args)


class Alignments:
    """Wrapper around the COM alignments info object."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def alignments_2d(self) -> list[Alignment2D]:
        """2D alignments."""
        return [Alignment2D(a) for a in self._com.Alignments2D]

    @property
    def alignments_3d(self) -> list[Alignment3D]:
        """3D alignments."""
        return [Alignment3D(a) for a in self._com.Alignments3D]

    @property
    def alignments_camera(self) -> list[AlignmentCamera]:
        """Camera alignments."""
        return [AlignmentCamera(a) for a in self._com.AlignmentsCamera]

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def type(self) -> Any:
        """Info-object type identifier."""
        return self._com.Type

    def get_video_rect(self) -> tuple[float, float, float, float]:
        """Gets the video rectangle defining the video coordinate system.

        Returns
        -------
        left : float
            The left side of the video rectangle
        top : float
            The top side of the video rectangle
        right : float
            The right side of the video rectangle
        bottom : float
            The bottom side of the video rectangle
        """
        return self._com.GetVideoRect()


class Element:
    """Wrapper around a COM geometry element."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def meas_point_indices(self) -> list[int]:
        """Measurement-point indices used by the element."""
        return [int(v) for v in self._com.MeasPointIndices]

    @property
    def meas_point_labels(self) -> list[Any]:
        """Measurement-point labels used by the element."""
        return list(self._com.MeasPointLabels)

    def meas_point_indices_by_status(
        self,
        status_mask: ScanStatus,
        status_compare: ScanStatus,
        any_bit: bool,
    ) -> list[int]:
        """Return measurement-point indices matching a scan-status mask."""
        return [
            int(v)
            for v in self._com.MeasPointIndicesByStatus(
                int(status_mask), int(status_compare), any_bit
            )
        ]

    def meas_point_labels_by_status(
        self,
        status_mask: ScanStatus,
        status_compare: ScanStatus,
        any_bit: bool,
    ) -> list[Any]:
        """Return measurement-point labels matching a scan-status mask."""
        return list(
            self._com.MeasPointLabelsByStatus(int(status_mask), int(status_compare), any_bit)
        )


class FrontEnd:
    """Wrapper around COM front-end information."""

    def __init__(self, obj: Any, *, expose_internals: bool = True) -> None:
        self._com = obj
        self._expose_internals = expose_internals

    @property
    def caps_value(self) -> int:
        """Front-end capabilities as a raw bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> FrontEndCaps:
        """Front-end capabilities as a decoded flag set."""
        return FrontEndCaps(self.caps_value)

    @property
    def firmware_version(self) -> str:
        """Firmware version of the front-end."""
        return self._com.FirmwareVersion

    @property
    def host_ip_address(self) -> str:
        """IP address of the connected host network adapter."""
        return self._com.HostIpAddress

    @property
    def mac_address(self) -> str:
        """MAC address of the front-end."""
        return self._com.MacAddress

    @property
    def name(self) -> str:
        """Front-end name."""
        return self._com.Name

    @property
    def type(self) -> FrontEndType:
        """Front-end type."""
        return FrontEndType(self._com.Type)

    @property
    def front_ends(self) -> Any:
        """Parent raw COM front-end collection."""
        return self._com.FrontEnds

    @property
    def control(self) -> Any:
        """Undocumented internal COM control object."""
        if not self._expose_internals:
            raise AttributeError(
                "FrontEnd.Control is undocumented; pass expose_internals=True " "to access it."
            )
        return self._com.Control

    @property
    def logical_channels(self) -> Any:
        """Undocumented internal COM logical-channel collection."""
        if not self._expose_internals:
            raise AttributeError(
                "FrontEnd.LogicalChannels is undocumented; pass "
                "expose_internals=True to access it."
            )
        return self._com.LogicalChannels

    @property
    def physical_channels(self) -> Any:
        """Undocumented internal COM physical-channel collection."""
        if not self._expose_internals:
            raise AttributeError(
                "FrontEnd.PhysicalChannels is undocumented; pass "
                "expose_internals=True to access it."
            )
        return self._com.PhysicalChannels


class Hardware:
    """Wrapper around COM hardware info."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def acq_board(self) -> AcqBoard:
        """Acquisition-board type used for the measurement."""
        return AcqBoard(self._com.AcqBoard)

    @property
    def acq_board_channel_count(self) -> int:
        """Number of acquisition-board channels."""
        return int(self._com.AcqBoardChannelCount)

    @property
    def acq_board_firmware_version(self) -> str:
        """Acquisition-board firmware version."""
        return self._com.AcqBoardFirmwareVersion

    @property
    def acq_board_max_frequency(self) -> float:
        """Maximum acquisition-board frequency."""
        return float(self._com.AcqBoardMaxFrequency)

    @property
    def acq_board_name(self) -> str:
        """Acquisition-board name."""
        return self._com.AcqBoardName

    @property
    def active_front_end(self) -> FrontEnd:
        """Active front-end used for the measurement."""
        return FrontEnd(self._com.ActiveFrontEnd)

    @property
    def coherence_optimizer_versions(self) -> list[str]:
        """Versions of system coherence optimizers."""
        return list(self._com.CoherenceOptimizerVersions)

    @property
    def front_end(self) -> FrontEndType:
        """Front-end type used for the measurement."""
        return FrontEndType(self._com.FrontEnd)

    @property
    def front_end_firmware_version(self) -> str:
        """Front-end firmware version."""
        return self._com.FrontEndFirmwareVersion

    @property
    def generator_channel_count(self) -> int:
        """Number of generator channels."""
        return int(self._com.GeneratorChannelCount)

    @property
    def generator_firmware_version(self) -> str:
        """Generator firmware version."""
        return self._com.GeneratorFirmwareVersion

    @property
    def generator_name(self) -> str:
        """Generator name."""
        return self._com.GeneratorName

    @property
    def generator_type(self) -> GeneratorType:
        """Generator type."""
        return GeneratorType(self._com.GeneratorType)

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def scan_head_type(self) -> ScanHeadType:
        """Scanning-head type used for the measurement."""
        return ScanHeadType(self._com.ScanHeadType)

    @property
    def sensor_head_names(self) -> list[str]:
        """Names of sensor heads used for the measurement."""
        return list(self._com.SensorHeadNames)

    @property
    def sensor_head_versions(self) -> list[str]:
        """Versions of sensor heads used for the measurement."""
        return list(self._com.SensorHeadVersions)

    @property
    def teds_sensors(self) -> Any:
        """TEDS sensor information."""
        return self._com.TedsSensors

    @property
    def type(self) -> Any:
        """Info-object type identifier."""
        return self._com.Type


class GeometryComponent:
    """Wrapper around a COM geometry component."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def components(self) -> list["GeometryComponent"]:
        """Child geometry components."""
        return [GeometryComponent(c) for c in self._com.Components]

    @property
    def description(self) -> str:
        """Geometry-component description."""
        return self._com.Description

    @property
    def id(self) -> Any:
        """Unique identifier of the geometry component."""
        return self._com.ID

    @property
    def indices(self) -> list[int]:
        """Measurement-point indices attached to the component."""
        return [int(v) for v in self._com.Indices]

    @property
    def labels(self) -> list[Any]:
        """Measurement-point labels attached to the component."""
        return list(self._com.Labels)

    @property
    def name(self) -> str:
        """Geometry-component name."""
        return self._com.Name


class MeasPoint:
    """Wrapper around a COM measurement point."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def coord_xyz(self) -> Any:
        """Retrieve the 3D coordinate of the measurement point."""
        return self._com.CoordXYZ(0, 0, 0)

    def set_coord_xyz(self, x: float, y: float, z: float) -> None:
        """Set the 3D coordinate of the measurement point."""
        self._com.SetCoordXYZ(x, y, z)

    def set_texture_xy_index(self, tex_x: float, tex_y: float, tex_index: int) -> None:
        """Set texture coordinates and texture index for the point."""
        self._com.SetTextureXYIndex(tex_x, tex_y, tex_index)

    def set_video_xy(self, video_x: float, video_y: float) -> None:
        """Set the video coordinate of the measurement point."""
        self._com.SetVideoXY(video_x, video_y)

    def texture_xy_index(self) -> Any:
        """Get texture coordinates and texture index for the point."""
        return self._com.TextureXYIndex(0, 0, 0)

    def toggle_valid_invalidated(self) -> None:
        """Toggle the invalidated state of the measurement point."""
        self._com.ToggleValidInvalidated()

    def video_xy(self) -> Any:
        """Retrieve the video coordinate of the measurement point."""
        return self._com.VideoXY(0, 0)

    @property
    def average_count(self) -> int:
        """Average count for the measurement point."""
        return int(self._com.AverageCount)

    @property
    def component(self) -> GeometryComponent:
        """Geometry component to which this point belongs."""
        return GeometryComponent(self._com.Component)

    @property
    def contained_file_active_id(self) -> Any:
        """Contained-file active identifier for combined files."""
        return self._com.ContainedFileActiveID

    @property
    def contained_file_active_mp_index(self) -> int:
        """Active contained-file measurement-point index."""
        return int(self._com.ContainedFileActiveMPIndex)

    @property
    def contained_file_index(self) -> int:
        """Contained-file index for combined files."""
        return int(self._com.ContainedFileIndex)

    @property
    def contained_file_meas_point_index(self) -> int:
        """Contained-file measurement-point index."""
        return int(self._com.ContainedFileMeasPointIndex)

    @property
    def focus_status(self) -> FocusStatus:
        """Laser-focus status flags."""
        return FocusStatus(int(self._com.FocusStatus))

    @property
    def focus_values(self) -> Any:
        """Laser-focus values."""
        return self._com.FocusValues

    @property
    def geometry_status(self) -> GeometryStatus:
        """3D geometry status flags."""
        return GeometryStatus(int(self._com.GeometryStatus))

    @property
    def index(self) -> int:
        """Measurement-point index."""
        return int(self._com.Index)

    @property
    def label(self) -> Any:
        """Measurement-point label."""
        return self._com.Label

    @property
    def min_sigma(self) -> float:
        """Minimum signal-enhancement sigma."""
        return float(self._com.MinSigma)

    @property
    def path_length_status(self) -> Any:
        """Optical path-length status."""
        return self._com.PathLengthStatus

    @property
    def path_length_value(self) -> Any:
        """Optical path-length value."""
        return self._com.PathLengthValue

    @property
    def scan_distances(self) -> list[float]:
        """Distances from the scanning head to the measurement point."""
        return [float(v) for v in self._com.ScanDistances]

    @property
    def scan_quantities_x(self) -> list[float]:
        """Scanner X values of the measurement point."""
        return [float(v) for v in self._com.ScanQuantitiesX]

    @property
    def scan_quantities_y(self) -> list[float]:
        """Scanner Y values of the measurement point."""
        return [float(v) for v in self._com.ScanQuantitiesY]

    @property
    def scan_quantity(self) -> PhysicalQuantity:
        """Physical quantity of the scanner values."""
        return PhysicalQuantity(self._com.ScanQuantity)

    @property
    def scan_status(self) -> ScanStatus:
        """Measurement status flags."""
        return ScanStatus(int(self._com.ScanStatus))

    @property
    def scan_status_disabled(self) -> bool:
        """Whether the scan-status disabled flag is set."""
        return bool(self._com.ScanStatusDisabled)

    @property
    def sigma(self) -> float:
        """Signal-enhancement sigma."""
        return float(self._com.Sigma)

    @property
    def vibrometer_range(self) -> Any:
        """Vibrometer range used for the point."""
        return self._com.VibrometerRange


class MeasPoints:
    """Pythonic wrapper around the COM measurement-point collection."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def __len__(self) -> int:
        """Number of measurement points."""
        return int(self._com.Count)

    def __getitem__(self, index: int) -> MeasPoint:
        """Return a measurement point by zero-based Python index."""
        count = len(self)
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise IndexError("MeasPoints index out of range")
        return MeasPoint(self._com.Item(index + 1))

    def __iter__(self):
        """Iterate over measurement points."""
        for i in range(len(self)):
            yield self[i]

    @property
    def average_sigma(self) -> float:
        """Average signal-enhancement sigma for all points."""
        return float(self._com.AverageSigma)

    @property
    def component(self) -> GeometryComponent:
        """Root geometry component."""
        return GeometryComponent(self._com.Component)

    @property
    def coordinates_to_array(self) -> np.ndarray:
        """3D coordinates for all points as an array of shape ``(n, 3)``."""
        return np.asarray(self._com.CoordinatesToArray).reshape((-1, 3))

    @property
    def count(self) -> int:
        """Number of measurement points."""
        return int(self._com.Count)

    @property
    def has_3d_coordinates(self) -> bool:
        """Whether the points are defined with 3D coordinates."""
        return bool(self._com.Has3DCoordinates)

    @property
    def has_non_optimal_points_to_invalidate(self) -> bool:
        """Whether non-optimal points exist that can be invalidated."""
        return bool(self._com.HasNonOptimalPointsToInvalidate)

    @property
    def has_sigma(self) -> bool:
        """Whether average signal-enhancement sigma is valid."""
        return bool(self._com.HasSigma)

    @property
    def is_3d_calculation_possible(self) -> bool:
        """Whether 3D coordinate calculation is possible."""
        return bool(self._com.Is3DCalculationPossible)

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def type(self) -> Any:
        """Info-object type identifier."""
        return self._com.Type

    @property
    def video_coordinates_to_array(self) -> np.ndarray:
        """Video coordinates for all points as an array of shape ``(n, 2)``."""
        return np.asarray(self._com.VideoCoordinatesToArray).reshape((-1, 2))

    def get_video_rect(self) -> tuple[float, float, float, float]:
        """Gets the video rectangle defining the video coordinate system.

        Returns
        -------
        left : float
            The left side of the video rectangle
        top : float
            The top side of the video rectangle
        right : float
            The right side of the video rectangle
        bottom : float
            The bottom side of the video rectangle
        """
        return self._com.GetVideoRect()


class MeasurementLocation:
    """Wrapper around a COM measurement-location object."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def channel_name(self) -> str:
        """Channel name."""
        return self._com.ChannelName

    @property
    def direction(self) -> Any:
        """Measurement direction or directions in object coordinates."""
        return self._com.Direction

    @property
    def meas_point_index(self) -> int:
        """One-based measurement-point index in the MeasPoints collection."""
        return int(self._com.MeasPointIndex)

    @property
    def mode(self) -> MeasurementLocationMode:
        """Measurement-location mode."""
        return MeasurementLocationMode(self._com.Mode)

    @property
    def normal(self) -> Any:
        """Normal vector or vectors in object coordinates."""
        return self._com.Normal

    @property
    def status(self) -> ScanStatus:
        """Measurement status of the referenced point."""
        return ScanStatus(int(self._com.Status))


class Profile:
    """Wrapper around a COM profile object."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def color(self) -> Any:
        """Profile color code."""
        return self._com.Color

    @property
    def lines(self) -> Any:
        """Profile line coordinates in video units."""
        return self._com.Lines

    @property
    def type(self) -> ProfileType:
        """Profile type."""
        return ProfileType(self._com.Type)


class DistanceSensorControl:
    """Wrapper around COM distance-sensor control."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def connected(self) -> bool:
        """Whether the geometry scan unit is connected."""
        return bool(self._com.Connected)

    @property
    def filter_on(self) -> bool:
        """Whether the geometry scan-unit filter is enabled."""
        return bool(self._com.FilterOn)

    @property
    def laser_on(self) -> bool:
        """Whether the geometry laser is on."""
        return bool(self._com.LaserOn)

    @property
    def scan_head_device(self) -> "ScanHeadDevice":
        """Associated scan-head device."""
        return ScanHeadDevice(self._com.ScanHeadDevice)

    @property
    def type(self) -> DistanceSensorType:
        """Distance-sensor type."""
        return DistanceSensorType(self._com.Type)


class ScanHeadLaserFocus:
    """Wrapper around COM scan-head focus settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def get_depth_of_focus_range(self, focus_position: float, depth_of_sharp_factor: float) -> Any:
        """Return the depth-of-focus range."""
        return self._com.GetDepthOfFocusRange(focus_position, depth_of_sharp_factor, 0, 0)

    @property
    def depth_of_focus_factor(self) -> float:
        """Depth-of-focus factor."""
        return float(self._com.DepthOfFocusFactor)

    @property
    def focal_length(self) -> float:
        """Objective focal length."""
        return float(self._com.FocalLength)

    @property
    def focus_command_time(self) -> float:
        """Time per focus command in seconds."""
        return float(self._com.FocusCommandTime)

    @property
    def focus_sharpness(self) -> float:
        """Focus sharpness."""
        return float(self._com.FocusSharpness)

    @property
    def focus_step_width(self) -> float:
        """Focus step width in meters."""
        return float(self._com.FocusStepWidth)

    @property
    def lens_to_mirror_distance(self) -> float:
        """Distance from objective to first mirror in meters."""
        return float(self._com.LensToMirrorDistance)

    @property
    def min_focus_distance(self) -> float:
        """Minimum focus distance in meters."""
        return float(self._com.MinFocusDistance)

    @property
    def scan_head_device(self) -> "ScanHeadDevice":
        """Associated scan-head device."""
        return ScanHeadDevice(self._com.ScanHeadDevice)

    @property
    def time_per_focus_step(self) -> float:
        """Time per focus step in seconds."""
        return float(self._com.TimePerFocusStep)


class ScanHeadScanner:
    """Wrapper around COM scan-head scanner settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def get_laser_origin(self) -> Any:
        """Return the laser origin at 0°/0° beam direction."""
        return self._com.GetLaserOrigin()

    def get_mirror_times_x(self, step: float) -> Any:
        """Return moving and waiting times for a horizontal step."""
        return self._com.GetMirrorTimesX(step)

    def get_mirror_times_y(self, step: float) -> Any:
        """Return moving and waiting times for a vertical step."""
        return self._com.GetMirrorTimesY(step)

    @property
    def cos_correction_active_x(self) -> bool:
        """Whether angle correction is active in the horizontal direction."""
        return bool(self._com.CosCorrectionActiveX)

    @property
    def cos_correction_active_y(self) -> bool:
        """Whether angle correction is active in the vertical direction."""
        return bool(self._com.CosCorrectionActiveY)

    @property
    def cos_correction_possible_x(self) -> bool:
        """Whether angle correction is possible in the horizontal direction."""
        return bool(self._com.CosCorrectionPossibleX)

    @property
    def cos_correction_possible_y(self) -> bool:
        """Whether angle correction is possible in the vertical direction."""
        return bool(self._com.CosCorrectionPossibleY)

    @property
    def inverted_x(self) -> bool:
        """Whether horizontal scanner polarity is inverted."""
        return bool(self._com.InvertedX)

    @property
    def inverted_y(self) -> bool:
        """Whether vertical scanner polarity is inverted."""
        return bool(self._com.InvertedY)

    @property
    def mirror_distance(self) -> float:
        """Distance between the two scanner mirrors in meters."""
        return float(self._com.MirrorDistance)

    @property
    def preferred_scan_direction(self) -> PreferredScanDirection:
        """Preferred scan direction."""
        return PreferredScanDirection(self._com.PreferredScanDirection)

    @property
    def quantity(self) -> PhysicalQuantity:
        """Physical quantity of the scanning head."""
        return PhysicalQuantity(self._com.Quantity)

    @property
    def quantity_max_x(self) -> float:
        """Maximum horizontal scanner value."""
        return float(self._com.QuantityMaxX)

    @property
    def quantity_max_y(self) -> float:
        """Maximum vertical scanner value."""
        return float(self._com.QuantityMaxY)

    @property
    def scan_head_device(self) -> "ScanHeadDevice":
        """Associated scan-head device."""
        return ScanHeadDevice(self._com.ScanHeadDevice)

    @property
    def swapped_xy(self) -> bool:
        """Whether horizontal and vertical scanners are swapped."""
        return bool(self._com.SwappedXY)


class ScanHeadVideo:
    """Wrapper around COM scan-head video settings."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def flipped_x(self) -> bool:
        """Whether the video image is flipped horizontally."""
        return bool(self._com.FlippedX)

    @property
    def flipped_y(self) -> bool:
        """Whether the video image is flipped vertically."""
        return bool(self._com.FlippedY)

    @property
    def scan_head_device(self) -> "ScanHeadDevice":
        """Associated scan-head device."""
        return ScanHeadDevice(self._com.ScanHeadDevice)

    @property
    def type_value(self) -> int:
        """Video type as a raw bitmask."""
        return int(self._com.Type)

    @property
    def type(self) -> ScanHeadVideoType:
        """Video type/capability flags."""
        return ScanHeadVideoType(self.type_value)


class ScanHeadDevice:
    """Wrapper around a COM scan-head device."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def caps_value(self) -> int:
        """Scanning-head capabilities as a raw bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> ScanHeadCaps:
        """Scanning-head capabilities as an ``IntFlag``."""
        return ScanHeadCaps(self.caps_value)

    @property
    def distance_sensor_control(self) -> DistanceSensorControl:
        """Distance-sensor control object."""
        return DistanceSensorControl(self._com.DistanceSensorControl)

    @property
    def focus(self) -> ScanHeadLaserFocus:
        """Scanning-head focus object."""
        return ScanHeadLaserFocus(self._com.Focus)

    @property
    def lens_magnification(self) -> float:
        """Objective magnification."""
        return float(self._com.LensMagnification)

    @property
    def pan_tilt_head(self) -> bool:
        """Whether a pan-tilt head is used."""
        return bool(self._com.PanTiltHead)

    @property
    def roll_angle(self) -> float:
        """Roll angle of the scanning head."""
        return float(self._com.RollAngle)

    @property
    def scanner(self) -> ScanHeadScanner:
        """Scanning-head scanner object."""
        return ScanHeadScanner(self._com.Scanner)

    @property
    def sensor_head_name(self) -> str:
        """Name of the active sensor head."""
        return self._com.SensorHeadName

    @property
    def sensor_head_names(self) -> list[str]:
        """All possible sensor-head names."""
        return list(self._com.SensorHeadNames)

    @property
    def video(self) -> ScanHeadVideo:
        """Scanning-head video object."""
        return ScanHeadVideo(self._com.Video)

    @property
    def scan_head_devices(self) -> list["ScanHeadDevice"]:
        """Parent scan-head-device collection as wrapped objects."""
        return [ScanHeadDevice(v) for v in self._com.ScanHeadDevices]


class ScanHeadDevicesInfo:
    """Wrapper around COM scan-head-devices info."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def scan_head_devices(self) -> list[ScanHeadDevice]:
        """Scan-head devices."""
        return [ScanHeadDevice(v) for v in self._com.ScanHeadDevices]

    @property
    def scan_head_name(self) -> str:
        """Scanning-head name."""
        return self._com.ScanHeadName

    @property
    def scan_head_type(self) -> ScanHeadType:
        """Scanning-head type."""
        return ScanHeadType(self._com.ScanHeadType)

    @property
    def type(self) -> Any:
        """Info-object type identifier."""
        return self._com.Type


class SpectrogramInfo:
    """Wrapper around COM spectrogram/campbell-diagram info."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def bandwidth(self) -> float:
        """Bandwidth used for spectrogram FFT calculation."""
        return float(self._com.Bandwidth)

    @property
    def bandwidth_from(self) -> float:
        """Lower bandwidth limit."""
        return float(self._com.BandwidthFrom)

    @property
    def bandwidth_to(self) -> float:
        """Upper bandwidth limit."""
        return float(self._com.BandwidthTo)

    @property
    def effective_sample_count(self) -> int:
        """Effective sample count used for FFT calculation."""
        return int(self._com.EffectiveSampleCount)

    @property
    def fft_count(self) -> int:
        """Number of FFTs in the resulting spectrogram."""
        return int(self._com.FFTCount)

    @property
    def fft_lines(self) -> int:
        """Number of FFT lines."""
        return int(self._com.FFTLines)

    @property
    def frequency_resolution(self) -> float:
        """Frequency resolution in hertz."""
        return float(self._com.FrequencyResolution)

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def overlap(self) -> float:
        """FFT overlap in percent."""
        return float(self._com.Overlap)

    @property
    def rpm_channel(self) -> str:
        """Name of the RPM channel."""
        return self._com.RPMChannel

    @property
    def sample_count(self) -> int:
        """Number of samples used to calculate an FFT."""
        return int(self._com.SampleCount)

    @property
    def sample_rate(self) -> float:
        """Time-trace sample rate in hertz."""
        return float(self._com.SampleRate)

    @property
    def sample_rate_bandwidth_ratio(self) -> float:
        """Ratio between sample rate and bandwidth."""
        return float(self._com.SampleRateBandwidthRatio)

    @property
    def time_from(self) -> float:
        """Start of the time range considered."""
        return float(self._com.TimeFrom)

    @property
    def time_to(self) -> float:
        """End of the time range considered."""
        return float(self._com.TimeTo)

    @property
    def total_sample_count(self) -> int:
        """Total number of samples available."""
        return int(self._com.TotalSampleCount)

    @property
    def type(self) -> Any:
        """Spectrogram type identifier."""
        return self._com.Type

    @property
    def used_fft_lines(self) -> int:
        """Number of used FFT lines."""
        return int(self._com.UsedFFTLines)

    @property
    def window_function(self) -> WindowFunction:
        """Window function used before FFT calculation."""
        return WindowFunction(self._com.WindowFunction)

    @property
    def window_parameter(self) -> Any:
        """Window-function parameter(s)."""
        return self._com.WindowParameter


class Texture:
    """Wrapper around a COM texture object."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def height(self) -> int:
        """Texture height in pixels."""
        return int(self._com.Height)

    @property
    def image(self) -> bytes:
        """Texture image bytes in BMP file format."""
        return bytes(self._com.Image)

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        return int(self._com.Width)


class VideoBitmap:
    """Wrapper around COM video-bitmap info."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def get_3d_video_offset(self) -> Any:
        """Get the video-position offset in 3D coordinates."""
        return self._com.Get3DVideoOffset()

    def get_image_rect(self) -> Any:
        """Get the image rectangle in video coordinates."""
        return self._com.GetImageRect()

    def get_video_offset(self) -> Any:
        """Get the video-position offset in video coordinates."""
        return self._com.GetVideoOffset()

    def load_image_file(self, file_name: str) -> None:
        """Load an image from the given file."""
        self._com.LoadImageFile(file_name)

    def save(self) -> None:
        """Commit changes to the underlying file."""
        self._com.Save()

    def set_image(self, image_bytes: bytes) -> None:
        """Set the image from an array of bytes."""
        self._com.SetImage(image_bytes)

    def image_bytes(
        self, fmt: GraphicFormatType = GraphicFormatType.BITMAP
    ) -> tuple[bytes, int, int]:
        """Return image bytes and dimensions.

        Parameters
        ----------
        fmt
            Graphic format to request from the COM interface.

        Returns
        -------
        tuple[bytes, int, int]
            Image bytes, width, and height.
        """
        data, width, height = self._com.Image(fmt.value)
        return bytes(data), int(width), int(height)

    def get_image(self) -> Image.Image:
        """Return a ``PIL.Image.Image`` object for the bitmap."""
        return Image.open(BytesIO(self.image_bytes()[0]))

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def texture_columns(self) -> int:
        """Number of columns in a stitched combined-file image."""
        return int(self._com.TextureColumns)

    @property
    def texture_coordinates(self) -> Any:
        """Texture coordinates for scan points."""
        return self._com.TextureCoordinates

    @property
    def texture_count(self) -> int:
        """Number of stitched video bitmaps."""
        return int(self._com.TextureCount)

    @property
    def texture_rows(self) -> int:
        """Number of rows in a stitched combined-file image."""
        return int(self._com.TextureRows)

    @property
    def type(self) -> Any:
        """Info-object type identifier."""
        return self._com.Type


class LensCalibration:
    """Wrapper around COM lens calibration information."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def caps_value(self) -> int:
        """Lens-calibration capabilities as a raw bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> LensCalibrationCaps:
        """Lens-calibration capabilities as an ``IntFlag``."""
        return LensCalibrationCaps(self.caps_value)

    @property
    def distance_x(self) -> float:
        """Horizontal calibration distance in meters."""
        return float(self._com.DistanceX)

    @property
    def distance_y(self) -> float:
        """Vertical calibration distance in meters."""
        return float(self._com.DistanceY)

    @property
    def magnification_factor_x(self) -> float:
        """Horizontal magnification factor."""
        return float(self._com.MagnificationFactorX)

    @property
    def magnification_factor_y(self) -> float:
        """Vertical magnification factor."""
        return float(self._com.MagnificationFactorY)

    @property
    def name(self) -> str:
        """Calibration name."""
        return self._com.Name

    @property
    def pixels_x(self) -> int:
        """Horizontal calibration distance in pixels."""
        return int(self._com.PixelsX)

    @property
    def pixels_y(self) -> int:
        """Vertical calibration distance in pixels."""
        return int(self._com.PixelsY)


class CameraSize:
    """Wrapper around COM camera-size information."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def chip_x(self) -> float:
        """Horizontal camera-chip size in meters."""
        return float(self._com.ChipX)

    @property
    def chip_y(self) -> float:
        """Vertical camera-chip size in meters."""
        return float(self._com.ChipY)

    @property
    def image_x(self) -> int:
        """Horizontal pixel count of the camera chip."""
        return int(self._com.ImageX)

    @property
    def image_y(self) -> int:
        """Vertical pixel count of the camera chip."""
        return int(self._com.ImageY)


class VideoMappingInfo:
    """Wrapper around COM video-mapping information."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def coord_xy_from_video(self, video_x: float, video_y: float) -> tuple[float, float]:
        """Retrieve world coordinates from video coordinates."""
        coord_x, coord_y = self._com.CoordXYFromVideo(video_x, video_y)
        return float(coord_x), float(coord_y)

    def video_xy_from_coord(self, coord_x: float, coord_y: float) -> tuple[float, float]:
        """Retrieve video coordinates from world coordinates."""
        video_x, video_y = self._com.VideoXYFromCoord(coord_x, coord_y)
        return float(video_x), float(video_y)

    def save(self) -> None:
        """Save modifications to the underlying file."""
        self._com.Save()

    @property
    def active_calibration(self) -> str:
        """Name of the active objective calibration."""
        return self._com.ActiveCalibration

    @property
    def calibrations(self) -> list[LensCalibration]:
        """Available lens calibrations."""
        return [LensCalibration(c) for c in self._com.Calibrations]

    @property
    def camera_size(self) -> CameraSize:
        """Camera-size object."""
        return CameraSize(self._com.CameraSize)

    @property
    def name(self) -> str:
        """Info-object name."""
        return self._com.Name

    @property
    def type(self) -> Any:
        """Info-object type identifier."""
        return self._com.Type


class XAxis:
    """Wrapper around a COM x-axis descriptor."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def max(self) -> float:
        """Maximum x-axis value."""
        return float(self._com.Max)

    @property
    def max_count(self) -> int:
        """Number of x-axis samples."""
        return int(self._com.MaxCount)

    @property
    def min(self) -> float:
        """Minimum x-axis value."""
        return float(self._com.Min)

    @property
    def name(self) -> str:
        """X-axis label."""
        return self._com.Name

    @property
    def unit(self) -> str:
        """X-axis unit."""
        return self._com.Unit


class YAxis:
    """Wrapper around a COM y-axis descriptor."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def max(self) -> float:
        """Maximum y-axis value."""
        return float(self._com.Max)

    @property
    def min(self) -> float:
        """Minimum y-axis value."""
        return float(self._com.Min)

    @property
    def name(self) -> str:
        """Y-axis label."""
        return self._com.Name

    @property
    def unit(self) -> str:
        """Y-axis unit."""
        return self._com.Unit


class Domain:
    """Wrapper around a COM domain."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def channels(self) -> list["Channel"]:
        """Channels in the domain."""
        return [Channel(c) for c in self._com.Channels]

    @property
    def name(self) -> str:
        """Domain name."""
        return self._com.Name

    @property
    def type(self) -> DomainType:
        """Domain type."""
        return DomainType(self._com.Type)


class Channel:
    """Wrapper around a COM signal channel."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def caps_value(self) -> int:
        """Channel capabilities as a raw bitmask."""
        return int(self._com.Caps)

    @property
    def caps(self) -> ChannelCaps:
        """Channel capabilities as an ``IntFlag``."""
        return ChannelCaps(self.caps_value)

    @property
    def caps_set(self) -> list[ChannelCaps]:
        """Individual channel capability flags that are set."""
        return _flags_list(self.caps_value, ChannelCaps)

    @property
    def domain(self) -> Domain:
        """Parent domain."""
        return Domain(self._com.Domain)

    @property
    def name(self) -> str:
        """Channel name."""
        return self._com.Name

    @property
    def signals(self) -> list["Signal"]:
        """Signals available on the channel."""
        return [Signal(s) for s in self._com.Signals]


class Signal:
    """Wrapper around a COM signal."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def channel(self) -> Channel:
        """Parent channel."""
        return Channel(self._com.Channel)

    @property
    def description(self) -> "SignalDescription":
        """Signal description."""
        return SignalDescription(self._com.Description)

    @property
    def displays(self) -> list["Display"]:
        """Available displays for the signal."""
        return [Display(d) for d in self._com.Displays]

    @property
    def name(self) -> str:
        """Signal name."""
        return self._com.Name


class Display:
    """Wrapper around a COM signal display."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def name(self) -> str:
        """Display name."""
        return self._com.Name

    @property
    def signal(self) -> Signal:
        """Parent signal."""
        return Signal(self._com.Signal)

    @property
    def type(self) -> DisplayType:
        """Display type."""
        return DisplayType(self._com.Type)


class Attribute:
    """Wrapper around a COM signal attribute."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def description(self) -> str:
        """Attribute description."""
        return self._com.Description

    @property
    def name(self) -> str:
        """Attribute name."""
        return self._com.Name

    @property
    def quantity(self) -> Any:
        """Physical quantity of the attribute value."""
        return self._com.Quantity

    @property
    def unit(self) -> str:
        """Unit of the attribute value."""
        return self._com.Unit

    @property
    def value(self) -> Any:
        """Attribute value."""
        return self._com.Value


class DegreeOfFreedomID:
    """Wrapper around a COM degree-of-freedom identifier."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def channel_name(self) -> str:
        """Channel name for the DOF."""
        return self._com.ChannelName

    @property
    def direction(self) -> DOFDirection:
        """DOF direction."""
        return DOFDirection(self._com.Direction)

    @property
    def node(self) -> Any:
        """DOF node number or identifier."""
        return self._com.Node

    @property
    def node_description(self) -> str:
        """Node description."""
        return self._com.NodeDescription

    @property
    def quantity(self) -> Any:
        """Physical quantity of the DOF channel."""
        return self._com.Quantity

    @property
    def unit(self) -> str:
        """Unit of the DOF channel."""
        return self._com.Unit


class SignalXAxis:
    """Wrapper around a COM signal x-axis descriptor."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def max(self) -> float:
        """Maximum x-axis value."""
        return float(self._com.Max)

    @property
    def max_count(self) -> int:
        """Number of x-axis samples."""
        return int(self._com.MaxCount)

    @property
    def min(self) -> float:
        """Minimum x-axis value."""
        return float(self._com.Min)

    @property
    def name(self) -> str:
        """X-axis label."""
        return self._com.Name

    @property
    def unit(self) -> str:
        """X-axis unit."""
        return self._com.Unit


class SignalYAxis:
    """Wrapper around a COM signal y-axis descriptor."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def max(self) -> float:
        """Maximum y-axis value."""
        return float(self._com.Max)

    @property
    def min(self) -> float:
        """Minimum y-axis value."""
        return float(self._com.Min)

    @property
    def name(self) -> str:
        """Y-axis label."""
        return self._com.Name

    @property
    def unit(self) -> str:
        """Y-axis unit."""
        return self._com.Unit


class SignalDescription:
    """Wrapper around COM signal metadata."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def attributes(self) -> list[Attribute]:
        """Additional attributes attached to the signal description."""
        return [Attribute(a) for a in self._com.Attributes]

    @property
    def complex(self) -> bool:
        """Whether the signal is complex-valued."""
        return bool(self._com.Complex)

    @property
    def data_type(self) -> DataType:
        """Signal data type."""
        return DataType(self._com.DataType)

    @property
    def db_reference(self) -> float:
        """Zero-dB reference value."""
        return float(self._com.DbReference)

    @property
    def domain_type(self) -> DomainType:
        """Domain type of the signal."""
        return DomainType(self._com.DomainType)

    @property
    def function_type(self) -> FunctionType:
        """Function type of the signal."""
        return FunctionType(self._com.FunctionType)

    @property
    def name(self) -> str:
        """Signal description name."""
        return self._com.Name

    @property
    def power_signal(self) -> bool:
        """Whether the signal is a power signal."""
        return bool(self._com.PowerSignal)

    @property
    def reference_dofs(self) -> list[DegreeOfFreedomID]:
        """Reference degree-of-freedom identifiers."""
        return [DegreeOfFreedomID(d) for d in self._com.ReferenceDOFs]

    @property
    def response_dofs(self) -> list[DegreeOfFreedomID]:
        """Response degree-of-freedom identifiers."""
        return [DegreeOfFreedomID(d) for d in self._com.ResponseDOFs]

    @property
    def rms_correction_factor(self) -> float:
        """RMS correction factor."""
        return float(self._com.RMSCorrectionFactor)

    @property
    def x_axis(self) -> SignalXAxis:
        """X-axis descriptor."""
        return SignalXAxis(self._com.XAxis)

    @property
    def y_axis(self) -> SignalYAxis:
        """Y-axis descriptor."""
        return SignalYAxis(self._com.YAxis)


class DataPoint:
    """Wrapper around a COM data point."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    @property
    def average_count(self) -> int:
        """Number of traces that contributed to this averaged result."""
        return int(self._com.AverageCount)

    @property
    def meas_point(self) -> MeasPoint:
        """Measurement point corresponding to this data point."""
        return MeasPoint(self._com.MeasPoint)

    @property
    def point_domain(self) -> "PointDomain":
        """Parent point domain."""
        return PointDomain(self._com.PointDomain)

    def get_3d_data(self, display: Any, frame: int = 0) -> np.ndarray:
        """Retrieve 3D data for a given display and frame."""
        disp_com = _unwrap_com(display)
        return np.asarray(self._com.Get3DData(disp_com, int(frame)))

    def get_3d_data_section(
        self,
        display: Any,
        frame: int,
        average: int,
        first_index: int,
        last_index: int,
    ) -> np.ndarray:
        """Retrieve a section of 3D data for a given display."""
        disp_com = _unwrap_com(display)
        return np.asarray(
            self._com.Get3DDataSection(
                disp_com,
                int(frame),
                int(average),
                int(first_index),
                int(last_index),
            )
        )

    def get_data(self, display: Any, frame: int = 0) -> np.ndarray:
        """Retrieve data for a given display and frame."""
        disp_com = _unwrap_com(display)
        return np.asarray(self._com.GetData(disp_com, int(frame)))

    def get_data_section(
        self,
        display: Any,
        frame: int,
        average: int,
        first_index: int,
        last_index: int,
    ) -> np.ndarray:
        """Retrieve a section of data for a given display."""
        disp_com = _unwrap_com(display)
        return np.asarray(
            self._com.GetDataSection(
                disp_com,
                int(frame),
                int(average),
                int(first_index),
                int(last_index),
            )
        )

    def get_frame_boundaries(self, display: Any, average: int = 1) -> np.ndarray:
        """Retrieve frame boundaries for a given time signal."""
        disp_com = _unwrap_com(display)
        return np.asarray(self._com.GetFrameBoundaries(disp_com, int(average)))

    def get_frames(self, display: Any) -> int:
        """Retrieve the frame count for a given display."""
        disp_com = _unwrap_com(display)
        return int(self._com.GetFrames(disp_com))

    def get_scan_status(self, display: Any) -> ScanStatus:
        """Retrieve the scan status for a given display."""
        disp_com = _unwrap_com(display)
        return ScanStatus(int(self._com.GetScanStatus(disp_com)))


class PointDomain:
    """Wrapper around a COM point domain."""

    def __init__(self, obj: Any) -> None:
        self._com = obj

    def get_x_axis(self) -> XAxis:
        """Return an x-axis descriptor for the point domain."""
        return XAxis(self._com.GetXAxis())

    def get_y_axes(self) -> list[YAxis]:
        """Return y-axis descriptors for the point domain."""
        return [YAxis(y) for y in self._com.GetYAxes()]

    @property
    def channels(self) -> list[Channel]:
        """Channels in the point domain."""
        return [Channel(c) for c in self._com.Channels]

    @property
    def data_points(self) -> list[DataPoint]:
        """Data points in the point domain."""
        return [DataPoint(dp) for dp in self._com.DataPoints]

    @property
    def name(self) -> str:
        """Point-domain name."""
        return self._com.Name

    @property
    def rms_correction_factor(self) -> float:
        """RMS correction factor for the point domain."""
        return float(self._com.RmsCorrectionFactor)

    @property
    def type(self) -> DomainType:
        """Point-domain type."""
        return DomainType(self._com.Type)


class PolyFile:
    """High-level wrapper around a Polytec PSV file COM object."""

    def __init__(self, obj_or_path: str | PathLike[str] | Any) -> None:
        """Open a Polytec file or wrap an existing COM file object.

        Parameters
        ----------
        obj_or_path
            Either a filesystem path to a Polytec file, or an existing COM file
            object returned by the PolyFile COM interface.

        Raises
        ------
        ModuleNotFoundError
            If ``pywin32`` is not installed.
        """
        if client is None:
            raise ModuleNotFoundError(
                "Could not import win32com.client. Run "
                "`python -m pip install pywin32` on a Windows operating system "
                "to install this package."
            )

        is_dispatch = False
        try:
            is_dispatch = isinstance(obj_or_path, client.CDispatch)
        except Exception:
            is_dispatch = False

        if is_dispatch or hasattr(obj_or_path, "Infos"):
            self._com = obj_or_path
        else:
            self._com = client.Dispatch("Polyfile.Polyfile")
            self._com.Open(str(obj_or_path))

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"PolyFile({self.file_name!r})"

    @property
    def contained_files(self) -> list["PolyFile"]:
        """Contained files if this file is a combined file.

        Raises
        ------
        ValueError
            If the file contains no embedded files.
        """
        try:
            count = int(self._com.ContainedFiles.Count)
        except Exception:
            count = len(self._com.ContainedFiles)

        if count == 0:
            raise ValueError("The present file contains no contained files.")
        return [PolyFile(obj.File) for obj in self._com.ContainedFiles]

    @property
    def parent_file(self) -> "PolyFile":
        """Parent combined file containing this file.

        Raises
        ------
        ValueError
            If this file has no parent combined file.
        """
        try:
            return PolyFile(self._com.CombinedFile)
        except com_error as e:
            raise ValueError("The present file contains no parent.") from e

    @property
    def title(self) -> str:
        """Document title."""
        return self._com.SummaryInfo.Title

    @property
    def comments(self) -> str:
        """Document comments."""
        return self._com.SummaryInfo.Comments

    @property
    def author(self) -> str:
        """Document author."""
        return self._com.SummaryInfo.Author

    @property
    def subject(self) -> str:
        """Document subject."""
        return self._com.SummaryInfo.Subject

    @property
    def date(self) -> Any:
        """File timestamp as returned by the COM interface."""
        return self._com.Version.FileTime

    @property
    def version(self) -> dict[str, Any]:
        """Version metadata reported by the COM interface."""
        keys = [
            "FileID",
            "FileTime",
            "FileVersion",
            "ProgramID",
            "ProgramVersion",
            "ProgramVersionBuild",
            "ProgramVersionMajor",
            "ProgramVersionMinor",
            "ProgramVersionRevision",
            "ProgramVersionString",
        ]
        return {key: getattr(self._com.Version, key) for key in keys}

    @property
    def file_name(self) -> str:
        """Path of the opened file."""
        return self._com.FileName

    @property
    def acquisition_mode(self) -> AcquisitionMode:
        """Active acquisition mode for the file."""
        return AcquisitionMode(self._com.Infos.AcquisitionInfoModes.ActiveMode)

    @property
    def acquisition_properties(self) -> dict[str, Any]:
        """Available acquisition-property wrappers for the active mode.

        Returns
        -------
        dict[str, Any]
            Mapping from snake_case property names to wrapper objects. Singular
            COM properties are returned as wrapper instances; collection-valued
            COM properties are returned as tuples of wrapper instances.
        """
        props = self._com.Infos.AcquisitionInfoModes.ActiveProperties

        spec = [
            ("average", "HasAverageProperties", "AverageProperties", AverageProperties, True),
            ("channels", "HasChannelsProperties", "ChannelsProperties", ChannelsProperties, False),
            (
                "fast_scans",
                "HasFastScansProperties",
                "FastScansProperties",
                FastScanProperties,
                True,
            ),
            ("fft", "HasFftProperties", "FftProperties", FFTProperties, True),
            ("front_end", "HasFrontEndProperties", "FrontEndProperties", FrontEndProperties, True),
            ("general", "HasGeneralProperties", "GeneralProperties", GeneralProperties, True),
            (
                "generators",
                "HasGeneratorsProperties",
                "GeneratorsProperties",
                GeneratorsProperties,
                False,
            ),
            (
                "multi_frame",
                "HasMultiFrameProperties",
                "MultiFrameProperties",
                MultiFrameProperties,
                True,
            ),
            (
                "signal_enhancement",
                "HasSignalEnhancementProperties",
                "SignalEnhancementProperties",
                SignalEnhancementProperties,
                True,
            ),
            ("time", "HasTimeProperties", "TimeProperties", TimeProperties, True),
            ("trigger", "HasTriggerProperties", "TriggerProperties", TriggerProperties, True),
            (
                "vibrometers",
                "HasVibrometersProperties",
                "VibrometersProperties",
                VibrometerProperties,
                False,
            ),
            ("zoom_fft", "HasZoomFftProperties", "ZoomFftProperties", ZoomFFTProperties, True),
        ]

        out: dict[str, Any] = {}
        for key, has_attr, com_attr, wrapper_cls, scalar in spec:
            if getattr(props, has_attr):
                com_value = getattr(props, com_attr)
                out[key] = (
                    wrapper_cls(com_value) if scalar else tuple(wrapper_cls(x) for x in com_value)
                )
        return out

    @property
    def alignments(self) -> Alignments:
        """Alignment information."""
        return Alignments(self._com.Infos.Alignments)

    @property
    def camera_settings(self) -> None:
        """Camera settings.

        Raises
        ------
        NotImplementedError
            The COM object returned by this property is undocumented and is not
            wrapped by this module.
        """
        raise NotImplementedError(
            "CameraSettings returns an ICamera object which is for internal use "
            "only and not documented."
        )

    @property
    def elements(self) -> list[Element]:
        """Geometry elements."""
        return [Element(val) for val in self._com.Infos.Elements]

    @property
    def hardware(self) -> Hardware:
        """Hardware information."""
        return Hardware(self._com.Infos.Hardware)

    @property
    def measurement_points(self) -> MeasPoints:
        """Measurement points."""
        return MeasPoints(self._com.Infos.MeasPoints)

    @property
    def measurement_locations(self) -> list[MeasurementLocation]:
        """Measurement locations.

        Raises
        ------
        ValueError
            If no measurement locations are defined in the dataset.
        """
        try:
            return [MeasurementLocation(v) for v in self._com.Infos.MeasurementLocations]
        except com_error as e:
            raise ValueError(
                "No measurement locations defined in this dataset. Did you mean "
                "`measurement_points`?"
            ) from e

    @property
    def profiles(self) -> list[Profile]:
        """Profiles defined in the dataset.

        Raises
        ------
        ValueError
            If no profiles are defined in the dataset.
        """
        try:
            return [Profile(v) for v in self._com.Infos.Profiles]
        except com_error as e:
            raise ValueError("No profiles defined in this dataset.") from e

    @property
    def scan_head_devices_info(self) -> ScanHeadDevicesInfo:
        """Scan-head-device information."""
        return ScanHeadDevicesInfo(self._com.Infos.ScanHeadDevicesInfo)

    @property
    def spectrogram_info(self) -> SpectrogramInfo:
        """Spectrogram information."""
        return SpectrogramInfo(self._com.Infos.SpectrogramInfo)

    @property
    def textures(self) -> list[Texture]:
        """Textures associated with the dataset."""
        return [Texture(v) for v in self._com.Infos.Textures]

    @property
    def video_bitmap(self) -> VideoBitmap:
        """Video bitmap information."""
        return VideoBitmap(self._com.Infos.VideoBitmap)

    @property
    def video_mapping_info(self) -> VideoMappingInfo:
        """Video mapping information."""
        return VideoMappingInfo(self._com.Infos.VideoMappingInfo)

    @property
    def point_domains(self) -> list[PointDomain]:
        """Available point domains."""
        return [PointDomain(v) for v in self._com.GetPointDomains()]

    def _get_single_point_domain(self, domain_type: DomainType) -> PointDomain:
        """Return the single point domain of the requested type.

        Raises
        ------
        ValueError
            If zero or more than one domains of the requested type are found.
        """
        domains = [domain for domain in self.point_domains if domain.type == domain_type]
        if len(domains) > 1:
            raise ValueError(f"Found more than one domain with type {domain_type.name}")
        if len(domains) == 0:
            raise ValueError(f"No data found with domain type {domain_type.name}")
        return domains[0]

    @staticmethod
    def _channel_is_reference(channel: Channel) -> bool:
        """Heuristic check for whether a channel is a reference channel.

        Notes
        -----
        This currently preserves the original module behavior by using the
        channel name. If a more reliable COM property becomes available, this
        method is the preferred place to update the logic.
        """
        return "vib" not in channel.name.lower()

    @staticmethod
    def _single_response_dof(
        description: SignalDescription, channel_name: str, signal_name: str
    ) -> DegreeOfFreedomID:
        """Return the single response DOF for a signal description."""
        if len(description.response_dofs) == 1:
            return description.response_dofs[0]
        raise NotImplementedError(
            f"Multiple Response DoFs were found for this signal "
            f"({channel_name} {signal_name}). Not sure how to proceed."
        )

    @staticmethod
    def _single_reference_dof(
        description: SignalDescription, channel_name: str, signal_name: str
    ) -> DegreeOfFreedomID:
        """Return the single reference DOF for a signal description."""
        if len(description.reference_dofs) == 1:
            return description.reference_dofs[0]
        raise NotImplementedError(
            f"Multiple Reference DoFs were found for this signal "
            f"({channel_name} {signal_name}). Not sure how to proceed."
        )

    @staticmethod
    def _single_response_or_reference_dof(
        description: SignalDescription, channel_name: str, signal_name: str
    ) -> DegreeOfFreedomID:
        """Return a single response DOF, or fall back to a single reference DOF."""
        if len(description.response_dofs) == 1:
            return description.response_dofs[0]
        if len(description.reference_dofs) == 1:
            return description.reference_dofs[0]
        raise NotImplementedError(
            f"Neither a single response DOF nor a single reference DOF could be "
            f"determined for signal ({channel_name} {signal_name})."
        )

    @staticmethod
    def _measpoint_coordinate_for_direction(data_point: DataPoint, direction_value: int) -> Any:
        """Build a coordinate-array entry for a measurement point."""
        return coordinate_array(data_point.meas_point.label, direction_value)

    @staticmethod
    def _append_comments(
        comment1s: list[str],
        comment2s: list[str],
        comment3s: list[str],
        comment4s: list[str],
        comment5s: list[str],
        *,
        unit: str,
        file_name: str,
        domain_name: str,
        channel_name: str,
        signal_name: str,
    ) -> None:
        """Append the standard comment fields used for sdynpy outputs."""
        comment1s.append(unit)
        comment2s.append(file_name[-80:])
        comment3s.append(domain_name)
        comment4s.append(channel_name)
        comment5s.append(signal_name)

    def get_transfer_functions(
        self,
        method: str = "h1",
        data_type: str = "acceleration",
        collapse_reference_signals: bool = True,
    ) -> TransferFunctionArray:
        """Extract transfer functions from the active spectrum point domain.

        Parameters
        ----------
        method
            Transfer-function estimator to extract. Supported values are
            ``"h1"`` and ``"h2"``.
        data_type
            Response quantity to extract from the signal name. Supported values
            are ``"acceleration"``, ``"velocity"``, and ``"displacement"``.
        collapse_reference_signals
            If ``True``, signals identified as reference-channel signals are
            averaged across measurement points and returned as a single trace
            per signal. If ``False``, all reference-channel traces are returned
            individually.

        Returns
        -------
        TransferFunctionArray
            An ``sdynpy`` transfer-function array produced by
            :func:`transfer_function_array`.
        """
        method_l = method.lower()
        data_type_l = data_type.lower()

        if method_l not in {"h1", "h2"}:
            raise ValueError('`method` argument must be one of "h1" or "h2"')
        if data_type_l not in {"acceleration", "velocity", "displacement"}:
            raise ValueError(
                '`data_type` argument must be one of "acceleration", ' '"velocity", "displacement"'
            )

        domain = self._get_single_point_domain(DomainType.SPECTRUM)
        data_points = domain.data_points

        abscissas: list[np.ndarray] = []
        response_coords: list[Any] = []
        reference_coords: list[Any] = []
        ordinates: list[np.ndarray] = []
        comment1s: list[str] = []
        comment2s: list[str] = []
        comment3s: list[str] = []
        comment4s: list[str] = []
        comment5s: list[str] = []

        for channel in domain.channels:
            if VERBOSE:
                print(f"Looking at Channel {channel.name}")

            for signal in channel.signals:
                if VERBOSE:
                    print(f"  Looking at Signal {signal.name}")

                signal_name_l = signal.name.lower()
                if method_l not in signal_name_l or data_type_l not in signal_name_l:
                    continue

                if VERBOSE:
                    print("    This signal is an FRF")

                display = _find_display_by_name(signal, "Real & Imag.")
                description = signal.description
                x_axis = description.x_axis
                abscissa = _build_linear_abscissa(x_axis)
                y_axis = description.y_axis

                response_dof = self._single_response_dof(description, channel.name, signal.name)
                reference_dof = self._single_reference_dof(description, channel.name, signal.name)

                response_coord = coordinate_array(response_dof.node, response_dof.direction.value)
                reference_coord = coordinate_array(
                    reference_dof.node, reference_dof.direction.value
                )

                if VERBOSE:
                    print("    Degrees of Freedom")
                    print(f"      Response: {response_coord}")
                    print(f"      Reference: {reference_coord}")

                this_ordinate: list[np.ndarray] = []
                this_measpoint_coords: list[Any] = []

                for data_point in data_points:
                    measpoint_coord = self._measpoint_coordinate_for_direction(
                        data_point, response_coord.direction
                    )
                    this_measpoint_coords.append(measpoint_coord)
                    data_array = data_point.get_data(display)
                    ordinate = data_array[::2] + 1j * data_array[1::2]
                    this_ordinate.append(ordinate)

                this_ordinate_arr = np.asarray(this_ordinate)

                if self._channel_is_reference(channel):
                    if VERBOSE:
                        print("    This is a reference channel.")
                    if collapse_reference_signals:
                        if VERBOSE:
                            print("    Collapsing Reference Signals to a single signal")
                        ordinates.append(np.mean(this_ordinate_arr, axis=0, keepdims=True))
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(response_coord[np.newaxis])
                        reference_coords.append(reference_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )
                    else:
                        if VERBOSE:
                            print("    Keeping all reference signals")
                        for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                            ordinates.append(ordinate[np.newaxis])
                            abscissas.append(abscissa[np.newaxis])
                            response_coords.append(response_coord[np.newaxis])
                            reference_coords.append(reference_coord[np.newaxis])
                            self._append_comments(
                                comment1s,
                                comment2s,
                                comment3s,
                                comment4s,
                                comment5s,
                                unit=y_axis.unit,
                                file_name=self.file_name,
                                domain_name=domain.name,
                                channel_name=f"{channel.name} at Measurement Point: {coord.node}",
                                signal_name=signal.name,
                            )
                else:
                    if VERBOSE:
                        print("    This is not a reference channel.")
                    for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                        ordinates.append(ordinate[np.newaxis])
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(coord[np.newaxis])
                        reference_coords.append(reference_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )

        if not ordinates:
            raise ValueError(
                f"No matching transfer-function signals were found for "
                f"method={method!r}, data_type={data_type!r}."
            )

        ordinate = np.concatenate(ordinates)
        abscissa = np.concatenate(abscissas)
        coordinate = np.concatenate(
            (
                np.concatenate(response_coords)[..., np.newaxis],
                np.concatenate(reference_coords)[..., np.newaxis],
            ),
            axis=-1,
        )
        return transfer_function_array(
            abscissa,
            ordinate,
            coordinate,
            comment1s,
            comment2s,
            comment3s,
            comment4s,
            comment5s,
        )

    def get_coherence(self, collapse_reference_signals: bool = True) -> CoherenceArray:
        """Extract coherence functions from the active spectrum point domain.

        Parameters
        ----------
        collapse_reference_signals
            If ``True``, signals identified as reference-channel signals are
            averaged across measurement points and returned as a single trace
            per signal. If ``False``, all traces are returned individually.

        Returns
        -------
        CoherenceArray
            An ``sdynpy`` coherence array produced by :func:`coherence_array`.
        """
        domain = self._get_single_point_domain(DomainType.SPECTRUM)
        data_points = domain.data_points

        abscissas: list[np.ndarray] = []
        response_coords: list[Any] = []
        reference_coords: list[Any] = []
        ordinates: list[np.ndarray] = []
        comment1s: list[str] = []
        comment2s: list[str] = []
        comment3s: list[str] = []
        comment4s: list[str] = []
        comment5s: list[str] = []

        for channel in domain.channels:
            if VERBOSE:
                print(f"Looking at Channel {channel.name}")

            for signal in channel.signals:
                if VERBOSE:
                    print(f"  Looking at Signal {signal.name}")

                if signal.name.lower() != "coherence":
                    continue

                if VERBOSE:
                    print("    This signal is coherence")

                display = _find_display_by_name(signal, "Magnitude")
                description = signal.description
                x_axis = description.x_axis
                abscissa = _build_linear_abscissa(x_axis)
                y_axis = description.y_axis

                response_dof = self._single_response_dof(description, channel.name, signal.name)
                reference_dof = self._single_reference_dof(description, channel.name, signal.name)

                response_coord = coordinate_array(response_dof.node, response_dof.direction.value)
                reference_coord = coordinate_array(
                    reference_dof.node, reference_dof.direction.value
                )

                if VERBOSE:
                    print("    Degrees of Freedom")
                    print(f"      Response: {response_coord}")
                    print(f"      Reference: {reference_coord}")

                this_ordinate: list[np.ndarray] = []
                this_measpoint_coords: list[Any] = []

                for data_point in data_points:
                    measpoint_coord = self._measpoint_coordinate_for_direction(
                        data_point, response_coord.direction
                    )
                    this_measpoint_coords.append(measpoint_coord)
                    ordinate = data_point.get_data(display)
                    this_ordinate.append(ordinate)

                this_ordinate_arr = np.asarray(this_ordinate)

                if self._channel_is_reference(channel):
                    if VERBOSE:
                        print("    This is a reference channel.")
                    if collapse_reference_signals:
                        if VERBOSE:
                            print("    Collapsing Reference Signals to a single signal")
                        ordinates.append(np.mean(this_ordinate_arr, axis=0, keepdims=True))
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(response_coord[np.newaxis])
                        reference_coords.append(reference_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )
                    else:
                        if VERBOSE:
                            print("    Keeping all reference signals")
                        for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                            ordinates.append(ordinate[np.newaxis])
                            abscissas.append(abscissa[np.newaxis])
                            response_coords.append(response_coord[np.newaxis])
                            reference_coords.append(reference_coord[np.newaxis])
                            self._append_comments(
                                comment1s,
                                comment2s,
                                comment3s,
                                comment4s,
                                comment5s,
                                unit=y_axis.unit,
                                file_name=self.file_name,
                                domain_name=domain.name,
                                channel_name=f"{channel.name} at Measurement Point: {coord.node}",
                                signal_name=signal.name,
                            )
                else:
                    if VERBOSE:
                        print("    This is not a reference channel.")
                    for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                        ordinates.append(ordinate[np.newaxis])
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(coord[np.newaxis])
                        reference_coords.append(reference_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )

        if not ordinates:
            raise ValueError("No coherence signals found in this dataset.")

        ordinate = np.concatenate(ordinates)
        abscissa = np.concatenate(abscissas)
        coordinate = np.concatenate(
            (
                np.concatenate(response_coords)[..., np.newaxis],
                np.concatenate(reference_coords)[..., np.newaxis],
            ),
            axis=-1,
        )
        return coherence_array(
            abscissa,
            ordinate,
            coordinate,
            comment1s,
            comment2s,
            comment3s,
            comment4s,
            comment5s,
        )

    def get_multiple_coherence(
        self, collapse_reference_signals: bool = True
    ) -> MultipleCoherenceArray:
        """Extract multiple-coherence functions from the active spectrum point domain.

        Parameters
        ----------
        collapse_reference_signals
            If ``True``, signals identified as reference-channel signals are
            averaged across measurement points and returned as a single trace
            per signal. If ``False``, all traces are returned individually.

        Returns
        -------
        MultipleCoherenceArray
            An ``sdynpy`` multiple-coherence array produced by
            :func:`multiple_coherence_array`.
        """
        domain = self._get_single_point_domain(DomainType.SPECTRUM)
        data_points = domain.data_points

        abscissas: list[np.ndarray] = []
        response_coords: list[Any] = []
        ordinates: list[np.ndarray] = []
        comment1s: list[str] = []
        comment2s: list[str] = []
        comment3s: list[str] = []
        comment4s: list[str] = []
        comment5s: list[str] = []

        for channel in domain.channels:
            if VERBOSE:
                print(f"Looking at Channel {channel.name}")

            for signal in channel.signals:
                if VERBOSE:
                    print(f"  Looking at Signal {signal.name}")

                if signal.name.lower() != "multiple coherence":
                    continue

                if VERBOSE:
                    print("    This signal is multiple coherence")

                display = _find_display_by_name(signal, "Magnitude")
                description = signal.description
                x_axis = description.x_axis
                abscissa = _build_linear_abscissa(x_axis)
                y_axis = description.y_axis

                response_dof = self._single_response_dof(description, channel.name, signal.name)
                response_coord = coordinate_array(response_dof.node, response_dof.direction.value)

                if VERBOSE:
                    print("    Degrees of Freedom")
                    print(f"      Response: {response_coord}")

                this_ordinate: list[np.ndarray] = []
                this_measpoint_coords: list[Any] = []

                for data_point in data_points:
                    measpoint_coord = self._measpoint_coordinate_for_direction(
                        data_point, response_coord.direction
                    )
                    this_measpoint_coords.append(measpoint_coord)
                    ordinate = data_point.get_data(display)
                    this_ordinate.append(ordinate)

                this_ordinate_arr = np.asarray(this_ordinate)

                if self._channel_is_reference(channel):
                    if VERBOSE:
                        print("    This is a reference channel.")
                    if collapse_reference_signals:
                        if VERBOSE:
                            print("    Collapsing Reference Signals to a single signal")
                        ordinates.append(np.mean(this_ordinate_arr, axis=0, keepdims=True))
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(response_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )
                    else:
                        if VERBOSE:
                            print("    Keeping all reference signals")
                        for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                            ordinates.append(ordinate[np.newaxis])
                            abscissas.append(abscissa[np.newaxis])
                            response_coords.append(response_coord[np.newaxis])
                            self._append_comments(
                                comment1s,
                                comment2s,
                                comment3s,
                                comment4s,
                                comment5s,
                                unit=y_axis.unit,
                                file_name=self.file_name,
                                domain_name=domain.name,
                                channel_name=f"{channel.name} at Measurement Point: {coord.node}",
                                signal_name=signal.name,
                            )
                else:
                    if VERBOSE:
                        print("    This is not a reference channel.")
                    for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                        ordinates.append(ordinate[np.newaxis])
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )

        if not ordinates:
            raise ValueError("No multiple coherence signals found in this dataset.")

        ordinate = np.concatenate(ordinates)
        abscissa = np.concatenate(abscissas)
        coordinate = np.concatenate(response_coords)[..., np.newaxis]
        return multiple_coherence_array(
            abscissa,
            ordinate,
            coordinate,
            comment1s,
            comment2s,
            comment3s,
            comment4s,
            comment5s,
        )

    def get_power_spectral_densities(
        self,
        data_type: str = "acceleration",
        collapse_reference_signals: bool = True,
    ) -> PowerSpectralDensityArray:
        """Extract power spectral densities from the active spectrum point domain.

        Parameters
        ----------
        data_type
            Preferred quantity to extract from the signal name. Supported values
            are ``"displacement"``, ``"velocity"``, and ``"acceleration"``.
        collapse_reference_signals
            If ``True``, signals identified as reference-channel signals are
            averaged across measurement points and returned as a single trace
            per signal. If ``False``, all traces are returned individually.

        Returns
        -------
        PowerSpectralDensityArray
            An ``sdynpy`` PSD array produced by
            :func:`power_spectral_density_array`.
        """
        data_type_l = data_type.lower()
        potential_data_types = {"displacement", "velocity", "acceleration"}
        if data_type_l not in potential_data_types:
            raise ValueError(
                '`data_type` argument must be one of "acceleration", "velocity", "displacement"'
            )

        domain = self._get_single_point_domain(DomainType.SPECTRUM)
        data_points = domain.data_points

        abscissas: list[np.ndarray] = []
        response_coords: list[Any] = []
        reference_coords: list[Any] = []
        ordinates: list[np.ndarray] = []
        comment1s: list[str] = []
        comment2s: list[str] = []
        comment3s: list[str] = []
        comment4s: list[str] = []
        comment5s: list[str] = []

        for channel in domain.channels:
            if VERBOSE:
                print(f"Looking at Channel {channel.name}")

            for signal in channel.signals:
                if VERBOSE:
                    print(f"  Looking at Signal {signal.name}")

                signal_name_l = signal.name.lower()
                if not (
                    "psd" in signal_name_l
                    and (
                        data_type_l in signal_name_l
                        or not any(dt in signal_name_l for dt in potential_data_types)
                    )
                ):
                    continue

                if VERBOSE:
                    print("    This signal is a PSD")

                display = _find_display_by_name(signal, "Magnitude")
                description = signal.description
                x_axis = description.x_axis
                abscissa = _build_linear_abscissa(x_axis)
                y_axis = description.y_axis

                response_dof = self._single_response_or_reference_dof(
                    description, channel.name, signal.name
                )
                response_coord = coordinate_array(response_dof.node, response_dof.direction.value)
                reference_coord = response_coord

                if VERBOSE:
                    print("    Degrees of Freedom")
                    print(f"      Response: {response_coord}")
                    print(f"      Reference: {reference_coord}")

                this_ordinate: list[np.ndarray] = []
                this_measpoint_coords: list[Any] = []

                for data_point in data_points:
                    measpoint_coord = self._measpoint_coordinate_for_direction(
                        data_point, response_coord.direction
                    )
                    this_measpoint_coords.append(measpoint_coord)
                    ordinate = data_point.get_data(display)
                    this_ordinate.append(ordinate)

                this_ordinate_arr = np.asarray(this_ordinate)

                if self._channel_is_reference(channel):
                    if VERBOSE:
                        print("    This is a reference channel.")
                    if collapse_reference_signals:
                        if VERBOSE:
                            print("    Collapsing Reference Signals to a single signal")
                        ordinates.append(np.mean(this_ordinate_arr, axis=0, keepdims=True))
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(response_coord[np.newaxis])
                        reference_coords.append(reference_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )
                    else:
                        if VERBOSE:
                            print("    Keeping all reference signals")
                        for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                            ordinates.append(ordinate[np.newaxis])
                            abscissas.append(abscissa[np.newaxis])
                            response_coords.append(response_coord[np.newaxis])
                            reference_coords.append(reference_coord[np.newaxis])
                            self._append_comments(
                                comment1s,
                                comment2s,
                                comment3s,
                                comment4s,
                                comment5s,
                                unit=y_axis.unit,
                                file_name=self.file_name,
                                domain_name=domain.name,
                                channel_name=f"{channel.name} at Measurement Point: {coord.node}",
                                signal_name=signal.name,
                            )
                else:
                    if VERBOSE:
                        print("    This is not a reference channel.")
                    for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                        ordinates.append(ordinate[np.newaxis])
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(coord[np.newaxis])
                        reference_coords.append(coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )

        if not ordinates:
            raise ValueError("No PSD signals found in this dataset.")

        ordinate = np.concatenate(ordinates)
        abscissa = np.concatenate(abscissas)
        coordinate = np.concatenate(
            (
                np.concatenate(response_coords)[..., np.newaxis],
                np.concatenate(reference_coords)[..., np.newaxis],
            ),
            axis=-1,
        )
        return power_spectral_density_array(
            abscissa,
            ordinate,
            coordinate,
            comment1s,
            comment2s,
            comment3s,
            comment4s,
            comment5s,
        )

    def get_time_response(self, collapse_reference_signals: bool = True) -> TimeHistoryArray:
        """Extract time histories from the active time-domain point domain.

        Parameters
        ----------
        collapse_reference_signals
            If ``True``, signals identified as reference-channel signals are
            averaged across measurement points and returned as a single trace
            per signal. If ``False``, all traces are returned individually.

        Returns
        -------
        TimeHistoryArray
            An ``sdynpy`` time-history array produced by
            :func:`time_history_array`.
        """
        domain = self._get_single_point_domain(DomainType.TIME)
        data_points = domain.data_points

        abscissas: list[np.ndarray] = []
        response_coords: list[Any] = []
        ordinates: list[np.ndarray] = []
        comment1s: list[str] = []
        comment2s: list[str] = []
        comment3s: list[str] = []
        comment4s: list[str] = []
        comment5s: list[str] = []

        for channel in domain.channels:
            if VERBOSE:
                print(f"Looking at Channel {channel.name}")

            for signal in channel.signals:
                if VERBOSE:
                    print(f"  Looking at Signal {signal.name}")

                if "voltage" in signal.name.lower():
                    continue

                if VERBOSE:
                    print("    This signal is time data")

                display = _find_display_by_name(signal, "Samples")
                description = signal.description
                x_axis = description.x_axis
                abscissa = _build_linear_abscissa(x_axis)
                y_axis = description.y_axis

                response_dof = self._single_response_or_reference_dof(
                    description, channel.name, signal.name
                )
                response_coord = coordinate_array(response_dof.node, response_dof.direction.value)

                if VERBOSE:
                    print("    Degrees of Freedom")
                    print(f"      Response: {response_coord}")

                this_ordinate: list[np.ndarray] = []
                this_measpoint_coords: list[Any] = []

                for data_point in data_points:
                    measpoint_coord = self._measpoint_coordinate_for_direction(
                        data_point, response_coord.direction
                    )
                    this_measpoint_coords.append(measpoint_coord)
                    ordinate = data_point.get_data(display)
                    this_ordinate.append(ordinate)

                this_ordinate_arr = np.asarray(this_ordinate)

                if self._channel_is_reference(channel):
                    if VERBOSE:
                        print("    This is a reference channel.")
                    if collapse_reference_signals:
                        if VERBOSE:
                            print("    Collapsing Reference Signals to a single signal")
                        ordinates.append(np.mean(this_ordinate_arr, axis=0, keepdims=True))
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(response_coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )
                    else:
                        if VERBOSE:
                            print("    Keeping all reference signals")
                        for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                            ordinates.append(ordinate[np.newaxis])
                            abscissas.append(abscissa[np.newaxis])
                            response_coords.append(response_coord[np.newaxis])
                            self._append_comments(
                                comment1s,
                                comment2s,
                                comment3s,
                                comment4s,
                                comment5s,
                                unit=y_axis.unit,
                                file_name=self.file_name,
                                domain_name=domain.name,
                                channel_name=f"{channel.name} at Measurement Point: {coord.node}",
                                signal_name=signal.name,
                            )
                else:
                    if VERBOSE:
                        print("    This is not a reference channel.")
                    for ordinate, coord in zip(this_ordinate_arr, this_measpoint_coords):
                        ordinates.append(ordinate[np.newaxis])
                        abscissas.append(abscissa[np.newaxis])
                        response_coords.append(coord[np.newaxis])
                        self._append_comments(
                            comment1s,
                            comment2s,
                            comment3s,
                            comment4s,
                            comment5s,
                            unit=y_axis.unit,
                            file_name=self.file_name,
                            domain_name=domain.name,
                            channel_name=channel.name,
                            signal_name=signal.name,
                        )

        if not ordinates:
            raise ValueError("No time-response signals found in this dataset.")

        ordinate = np.concatenate(ordinates)
        abscissa = np.concatenate(abscissas)
        coordinate = np.concatenate(response_coords)[..., np.newaxis]
        return time_history_array(
            abscissa,
            ordinate,
            coordinate,
            comment1s,
            comment2s,
            comment3s,
            comment4s,
            comment5s,
        )

    def get_geometry(self) -> Geometry:
        """Build an ``sdynpy`` geometry object from measurement points and elements.

        Measurement-point coordinates are converted into nodes, and simple
        element connectivity is inferred from element point-label counts:
        2-point elements become tracelines, 3-point elements become type 61
        elements, and 4-point elements become type 64 elements.
        """
        coordinate_system = coordinate_system_array(1)
        coords = self.measurement_points.coordinates_to_array
        node_ids = [pt.label for pt in self.measurement_points]
        nodes = node_array(node_ids, coords)
        geometry = Geometry(nodes, coordinate_system)

        for element in self.elements:
            labels = element.meas_point_labels
            if len(labels) == 2:
                geometry.add_traceline(labels)
            elif len(labels) == 3:
                geometry.add_element(61, labels)
            elif len(labels) == 4:
                geometry.add_element(64, labels)

        return geometry

    def _get_contained_files_or_none(self) -> list["PolyFile"] | None:
        """Return contained files if this is a combined file, else ``None``."""
        try:
            contained = self.contained_files
        except ValueError:
            return None
        return contained if contained else None

    def _delegate_image_method_to_contained_files(
        self,
        method_name: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Call an image-producing method recursively on contained files."""
        contained = self._get_contained_files_or_none()
        if contained is None:
            raise ValueError(
                "_delegate_image_method_to_contained_files called on a file "
                "without contained files."
            )

        kwargs = dict(kwargs)
        if "ax" in kwargs:
            kwargs["ax"] = None

        out: list[Any] = []
        for child in contained:
            method = getattr(child, method_name)
            out.append(method(*args, **kwargs))
        return out

    def get_image(self) -> Image.Image | list[Any]:
        """Return the stored test image as a PIL image.

        For combined files, returns a list of images from the contained files.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files("get_image")

        return self.video_bitmap.get_image().copy()

    @staticmethod
    def _pil_color(color: str) -> str:
        """Convert Matplotlib-style short color codes to PIL-compatible names."""
        color_map = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            "w": "white",
        }
        return color_map.get(color, color)

    @staticmethod
    def _video_to_pixel(
        x: float,
        y: float,
        video_rect: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        """Convert Polytec video coordinates to image pixel coordinates."""
        left, top, right, bottom = video_rect
        x_pixel = (x - left) * image_width / (right - left)
        y_pixel = (y - top) * image_height / (bottom - top)
        return x_pixel, y_pixel

    @staticmethod
    def _default_alignment_colors() -> list[str]:
        """Default color cycle for plotting multiple alignments."""
        return ["red", "cyan", "yellow", "magenta", "lime", "orange", "white"]

    @staticmethod
    def _default_alignment_markers() -> list[str]:
        """Default marker cycle for plotting multiple alignments."""
        return ["o", "x", "^", "s", "d", "+", "*"]

    @staticmethod
    def _default_status_colors() -> list[str]:
        """Default color cycle for status plotting."""
        return [
            "red",
            "cyan",
            "yellow",
            "magenta",
            "lime",
            "orange",
            "white",
            "deepskyblue",
            "springgreen",
            "gold",
        ]

    @staticmethod
    def _default_status_markers() -> list[str]:
        """Default marker cycle for status plotting."""
        return ["o", "x", "^", "s", "d", "+", "*", "v", "<", ">"]

    @staticmethod
    def _decode_flag_label(value: int, enum_cls: type[IntFlag]) -> str:
        """Convert a raw IntFlag value into a readable label."""
        try:
            flags = enum_cls(int(value))
        except Exception:
            return str(value)

        if int(value) == 0:
            zero_member = None
            for member in enum_cls:
                if int(member.value) == 0:
                    zero_member = member.name
                    break
            return zero_member or "NONE"

        names: list[str] = []
        for member in enum_cls:
            if int(member.value) == 0:
                continue
            if flags & member:
                names.append(member.name)

        return " | ".join(names) if names else str(int(value))

    def _plot_points_on_image(
        self,
        points_xy: list[tuple[float, float]],
        *,
        image: Image.Image | None = None,
        video_rect: tuple[float, float, float, float] | None = None,
        mode: str = "matplotlib",
        color: str = "red",
        marker: str = "o",
        size: float = 40.0,
        linewidth: float = 2.0,
        show_labels: bool = False,
        labels: list[str] | None = None,
        label_color: str | None = None,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot video-coordinate points on the stored image."""
        if image is None:
            image = self.get_image()
            if isinstance(image, list):
                raise ValueError(
                    "Combined files return multiple images. This method should "
                    "only be called on a leaf file."
                )
        else:
            image = image.copy()

        if video_rect is None:
            raise ValueError("`video_rect` must be provided.")

        image_width, image_height = image.size
        pixel_points = [
            self._video_to_pixel(x, y, video_rect, image_width, image_height) for x, y in points_xy
        ]

        if label_color is None:
            label_color = color

        if mode.lower() == "pil":
            draw = ImageDraw.Draw(image)
            radius = max(2.0, float(linewidth) + 2.0)
            pil_color = self._pil_color(color)
            pil_label_color = self._pil_color(label_color)

            for i, (xp, yp) in enumerate(pixel_points):
                draw.ellipse(
                    (xp - radius, yp - radius, xp + radius, yp + radius),
                    outline=pil_color,
                    width=max(1, int(round(linewidth))),
                )
                if show_labels and labels is not None and i < len(labels):
                    draw.text(
                        (xp + radius + 2, yp + radius + 2),
                        str(labels[i]),
                        fill=pil_label_color,
                    )
            return image

        if mode.lower() == "matplotlib":

            created_fig = False
            if ax is None:
                fig, ax = plt.subplots()
                created_fig = True
            else:
                fig = ax.figure

            if not ax.images:
                ax.imshow(image)

            xs = [p[0] for p in pixel_points]
            ys = [p[1] for p in pixel_points]
            ax.scatter(xs, ys, c=color, marker=marker, s=size)

            if show_labels and labels is not None:
                for (xp, yp), label in zip(pixel_points, labels):
                    ax.text(xp, yp, str(label), color=label_color)

            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")

            if title is not None:
                ax.set_title(title)

            if created_fig:
                fig.tight_layout()

            return fig, ax

        raise ValueError("`mode` must be either 'pil' or 'matplotlib'.")

    def get_measpoint_image(
        self,
        *,
        mode: str = "matplotlib",
        color: str = "red",
        marker: str = "o",
        size: float = 40.0,
        linewidth: float = 2.0,
        show_labels: bool = False,
        use_point_labels: bool = True,
        ax: Any = None,
        title: str | None = "Measurement Points",
    ) -> Any:
        """Plot measurement-point locations on the stored image.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_measpoint_image",
                mode=mode,
                color=color,
                marker=marker,
                size=size,
                linewidth=linewidth,
                show_labels=show_labels,
                use_point_labels=use_point_labels,
                ax=ax,
                title=title,
            )

        image = self.get_image()
        video_rect = tuple(float(v) for v in self.measurement_points.get_video_rect())

        points_xy: list[tuple[float, float]] = []
        labels: list[str] = []

        for pt in self.measurement_points:
            try:
                x, y = pt.video_xy()
            except Exception:
                continue
            points_xy.append((float(x), float(y)))
            labels.append(str(pt.label if use_point_labels else pt.index))

        return self._plot_points_on_image(
            points_xy,
            image=image,
            video_rect=video_rect,
            mode=mode,
            color=color,
            marker=marker,
            size=size,
            linewidth=linewidth,
            show_labels=show_labels,
            labels=labels,
            ax=ax,
            title=title,
        )

    def get_alignment2d_image(
        self,
        alignment_index: int | None = None,
        *,
        mode: str = "matplotlib",
        color: str | None = None,
        marker: str | None = None,
        size: float = 60.0,
        linewidth: float = 2.0,
        show_labels: bool = True,
        ax: Any = None,
        title: str | None = "2D Alignment Points",
    ) -> Any:
        """Plot 2D alignment points on the stored image.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_alignment2d_image",
                alignment_index,
                mode=mode,
                color=color,
                marker=marker,
                size=size,
                linewidth=linewidth,
                show_labels=show_labels,
                ax=ax,
                title=title,
            )

        alignments = self.alignments.alignments_2d
        if alignment_index is None:
            selected_alignments = list(enumerate(alignments))
        else:
            selected_alignments = [(alignment_index, alignments[alignment_index])]

        image = self.get_image()
        video_rect = tuple(float(v) for v in self.alignments.get_video_rect())

        default_colors = self._default_alignment_colors()
        default_markers = self._default_alignment_markers()

        if mode.lower() == "pil":
            out_image = image.copy()
            for i_align, alignment in selected_alignments:
                this_color = (
                    color if color is not None else default_colors[i_align % len(default_colors)]
                )
                this_marker = (
                    marker
                    if marker is not None
                    else default_markers[i_align % len(default_markers)]
                )

                points_xy = [
                    (float(pt.video_x), float(pt.video_y)) for pt in alignment.align_2d_points
                ]
                labels = [f"A{i_align}:{i}" for i in range(len(points_xy))]

                out_image = self._plot_points_on_image(
                    points_xy,
                    image=out_image,
                    video_rect=video_rect,
                    mode="pil",
                    color=this_color,
                    marker=this_marker,
                    size=size,
                    linewidth=linewidth,
                    show_labels=show_labels,
                    labels=labels,
                    title=None,
                )
            return out_image

        if mode.lower() == "matplotlib":
            created_fig = False
            if ax is None:
                fig, ax = plt.subplots()
                created_fig = True
                ax.imshow(image)
            else:
                fig = ax.figure
                if not ax.images:
                    ax.imshow(image)

            image_width, image_height = image.size

            for i_align, alignment in selected_alignments:
                this_color = (
                    color if color is not None else default_colors[i_align % len(default_colors)]
                )
                this_marker = (
                    marker
                    if marker is not None
                    else default_markers[i_align % len(default_markers)]
                )

                points_xy = [
                    (float(pt.video_x), float(pt.video_y)) for pt in alignment.align_2d_points
                ]
                pixel_points = [
                    self._video_to_pixel(x, y, video_rect, image_width, image_height)
                    for x, y in points_xy
                ]
                xs = [p[0] for p in pixel_points]
                ys = [p[1] for p in pixel_points]

                label = f"2D Alignment {i_align}" if alignment_index is None else "2D Alignment"
                ax.scatter(xs, ys, c=this_color, marker=this_marker, s=size, label=label)

                if show_labels:
                    for j, (xp, yp) in enumerate(pixel_points):
                        ax.text(xp, yp, f"A{i_align}:{j}", color=this_color)

            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")

            if title is not None:
                ax.set_title(title)

            if alignment_index is None and len(selected_alignments) > 1:
                ax.legend()

            if created_fig:
                fig.tight_layout()

            return fig, ax

        raise ValueError("`mode` must be either 'pil' or 'matplotlib'.")

    def get_alignment3d_image(
        self,
        alignment_index: int | None = None,
        *,
        mode: str = "matplotlib",
        color: str | None = None,
        marker: str | None = None,
        size: float = 60.0,
        linewidth: float = 2.0,
        show_labels: bool = True,
        use_point_labels: bool = True,
        ax: Any = None,
        title: str | None = "3D Alignment Points",
    ) -> Any:
        """Plot 3D alignment points on the stored image.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_alignment3d_image",
                alignment_index,
                mode=mode,
                color=color,
                marker=marker,
                size=size,
                linewidth=linewidth,
                show_labels=show_labels,
                use_point_labels=use_point_labels,
                ax=ax,
                title=title,
            )

        alignments = self.alignments.alignments_3d
        if alignment_index is None:
            selected_alignments = list(enumerate(alignments))
        else:
            selected_alignments = [(alignment_index, alignments[alignment_index])]

        image = self.get_image()
        video_rect = tuple(float(v) for v in self.alignments.get_video_rect())

        default_colors = self._default_alignment_colors()
        default_markers = self._default_alignment_markers()

        if mode.lower() == "pil":
            out_image = image.copy()
            for i_align, alignment in selected_alignments:
                this_color = (
                    color if color is not None else default_colors[i_align % len(default_colors)]
                )
                this_marker = (
                    marker
                    if marker is not None
                    else default_markers[i_align % len(default_markers)]
                )

                points_xy = [
                    (float(pt.video_x), float(pt.video_y)) for pt in alignment.align_3d_points
                ]
                if use_point_labels:
                    labels = [str(pt.label) for pt in alignment.align_3d_points]
                else:
                    labels = [f"A{i_align}:{i}" for i in range(len(points_xy))]

                out_image = self._plot_points_on_image(
                    points_xy,
                    image=out_image,
                    video_rect=video_rect,
                    mode="pil",
                    color=this_color,
                    marker=this_marker,
                    size=size,
                    linewidth=linewidth,
                    show_labels=show_labels,
                    labels=labels,
                    title=None,
                )
            return out_image

        if mode.lower() == "matplotlib":
            created_fig = False
            if ax is None:
                fig, ax = plt.subplots()
                created_fig = True
                ax.imshow(image)
            else:
                fig = ax.figure
                if not ax.images:
                    ax.imshow(image)

            image_width, image_height = image.size

            for i_align, alignment in selected_alignments:
                this_color = (
                    color if color is not None else default_colors[i_align % len(default_colors)]
                )
                this_marker = (
                    marker
                    if marker is not None
                    else default_markers[i_align % len(default_markers)]
                )

                points_xy = [
                    (float(pt.video_x), float(pt.video_y)) for pt in alignment.align_3d_points
                ]
                pixel_points = [
                    self._video_to_pixel(x, y, video_rect, image_width, image_height)
                    for x, y in points_xy
                ]
                xs = [p[0] for p in pixel_points]
                ys = [p[1] for p in pixel_points]

                label = f"3D Alignment {i_align}" if alignment_index is None else "3D Alignment"
                ax.scatter(xs, ys, c=this_color, marker=this_marker, s=size, label=label)

                if show_labels:
                    if use_point_labels:
                        point_labels = [str(pt.label) for pt in alignment.align_3d_points]
                    else:
                        point_labels = [f"A{i_align}:{j}" for j in range(len(pixel_points))]
                    for (xp, yp), point_label in zip(pixel_points, point_labels):
                        ax.text(xp, yp, point_label, color=this_color)

            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")

            if title is not None:
                ax.set_title(title)

            if alignment_index is None and len(selected_alignments) > 1:
                ax.legend()

            if created_fig:
                fig.tight_layout()

            return fig, ax

        raise ValueError("`mode` must be either 'pil' or 'matplotlib'.")

    def _get_status_image(
        self,
        *,
        status_attr: str,
        enum_cls: type[IntFlag],
        mode: str = "matplotlib",
        size: float = 60.0,
        linewidth: float = 2.0,
        show_labels: bool = False,
        use_point_labels: bool = True,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Plot measurement points grouped by status flags."""
        image = self.get_image()
        if isinstance(image, list):
            raise ValueError(
                "Combined files should be handled by the public status-image "
                "methods before calling _get_status_image."
            )

        video_rect = tuple(float(v) for v in self.measurement_points.get_video_rect())

        grouped_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
        grouped_labels: dict[int, list[str]] = defaultdict(list)

        for pt in self.measurement_points:
            try:
                x, y = pt.video_xy()
            except Exception:
                continue

            status_value = int(getattr(pt, status_attr))
            grouped_points[status_value].append((float(x), float(y)))
            grouped_labels[status_value].append(str(pt.label if use_point_labels else pt.index))

        if not grouped_points:
            raise ValueError(f"No points with plottable {status_attr} data were found.")

        unique_status_values = sorted(grouped_points.keys())

        colors = self._default_status_colors()
        markers = self._default_status_markers()

        if mode.lower() == "pil":
            out_image = image.copy()
            for i_status, status_value in enumerate(unique_status_values):
                this_color = colors[i_status % len(colors)]
                this_marker = markers[i_status % len(markers)]
                labels = grouped_labels[status_value] if show_labels else None

                out_image = self._plot_points_on_image(
                    grouped_points[status_value],
                    image=out_image,
                    video_rect=video_rect,
                    mode="pil",
                    color=this_color,
                    marker=this_marker,
                    size=size,
                    linewidth=linewidth,
                    show_labels=show_labels,
                    labels=labels,
                    title=None,
                )
            return out_image

        if mode.lower() == "matplotlib":
            created_fig = False
            if ax is None:
                fig, ax = plt.subplots()
                created_fig = True
                ax.imshow(image)
            else:
                fig = ax.figure
                if not ax.images:
                    ax.imshow(image)

            image_width, image_height = image.size

            for i_status, status_value in enumerate(unique_status_values):
                this_color = colors[i_status % len(colors)]
                this_marker = markers[i_status % len(markers)]

                pixel_points = [
                    self._video_to_pixel(x, y, video_rect, image_width, image_height)
                    for x, y in grouped_points[status_value]
                ]
                xs = [p[0] for p in pixel_points]
                ys = [p[1] for p in pixel_points]

                legend_label = self._decode_flag_label(status_value, enum_cls)

                ax.scatter(
                    xs,
                    ys,
                    c=this_color,
                    marker=this_marker,
                    s=size,
                    label=legend_label,
                )

                if show_labels:
                    point_labels = grouped_labels[status_value]
                    for (xp, yp), point_label in zip(pixel_points, point_labels):
                        ax.text(xp, yp, point_label, color=this_color)

            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
            ax.set_aspect("equal")
            ax.set_xlabel("Pixel X")
            ax.set_ylabel("Pixel Y")

            if title is not None:
                ax.set_title(title)

            ax.legend()

            if created_fig:
                fig.tight_layout()

            return fig, ax

        raise ValueError("`mode` must be either 'pil' or 'matplotlib'.")

    def get_geometry_status_image(
        self,
        *,
        mode: str = "matplotlib",
        size: float = 60.0,
        linewidth: float = 2.0,
        show_labels: bool = False,
        use_point_labels: bool = True,
        ax: Any = None,
        title: str | None = "Geometry Status",
    ) -> Any:
        """Plot measurement points grouped by geometry status.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_geometry_status_image",
                mode=mode,
                size=size,
                linewidth=linewidth,
                show_labels=show_labels,
                use_point_labels=use_point_labels,
                ax=ax,
                title=title,
            )

        return self._get_status_image(
            status_attr="geometry_status",
            enum_cls=GeometryStatus,
            mode=mode,
            size=size,
            linewidth=linewidth,
            show_labels=show_labels,
            use_point_labels=use_point_labels,
            ax=ax,
            title=title,
        )

    def get_focus_status_image(
        self,
        *,
        mode: str = "matplotlib",
        size: float = 60.0,
        linewidth: float = 2.0,
        show_labels: bool = False,
        use_point_labels: bool = True,
        ax: Any = None,
        title: str | None = "Focus Status",
    ) -> Any:
        """Plot measurement points grouped by focus status.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_focus_status_image",
                mode=mode,
                size=size,
                linewidth=linewidth,
                show_labels=show_labels,
                use_point_labels=use_point_labels,
                ax=ax,
                title=title,
            )

        return self._get_status_image(
            status_attr="focus_status",
            enum_cls=FocusStatus,
            mode=mode,
            size=size,
            linewidth=linewidth,
            show_labels=show_labels,
            use_point_labels=use_point_labels,
            ax=ax,
            title=title,
        )

    def get_scan_status_image(
        self,
        *,
        mode: str = "matplotlib",
        size: float = 60.0,
        linewidth: float = 2.0,
        show_labels: bool = False,
        use_point_labels: bool = True,
        ax: Any = None,
        title: str | None = "Scan Status",
    ) -> Any:
        """Plot measurement points grouped by scan status.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_scan_status_image",
                mode=mode,
                size=size,
                linewidth=linewidth,
                show_labels=show_labels,
                use_point_labels=use_point_labels,
                ax=ax,
                title=title,
            )

        return self._get_status_image(
            status_attr="scan_status",
            enum_cls=ScanStatus,
            mode=mode,
            size=size,
            linewidth=linewidth,
            show_labels=show_labels,
            use_point_labels=use_point_labels,
            ax=ax,
            title=title,
        )

    def get_alignment3d_quality(self):
        """Returns the alignment quality for each laser head.

        If the file contains multiple files, then one quality score
        will be returned for each contained file.

        Returns
        -------
        quality : np.ndarray
            A 1D array for single-scan files and a 2D array for combined files.
            Each row in the 2D array represents a different scan.
        """
        contained = self._get_contained_files_or_none()
        if contained:
            quality = []
            for file in self.contained_files:
                quality.append(
                    [alignment.current_quality for alignment in file.alignments.alignments_3d]
                    )
        else:
            quality = [alignment.current_quality for alignment in self.alignments.alignments_3d]
        return np.array(quality)

    @staticmethod
    def _pixel_to_video(
        x_pixel: float,
        y_pixel: float,
        video_rect: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        """Convert image pixel coordinates to Polytec video coordinates."""
        left, top, right, bottom = video_rect
        x_video = left + x_pixel * (right - left) / image_width
        y_video = top + y_pixel * (bottom - top) / image_height
        return x_video, y_video

    def pixel_to_video(
        self,
        x_pixel: float,
        y_pixel: float,
    ) -> tuple[float, float] | list[Any]:
        """Convert image pixel coordinates to Polytec video coordinates.

        For combined files, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "pixel_to_video",
                x_pixel,
                y_pixel,
            )

        image = self.get_image()
        if isinstance(image, list):
            raise ValueError(
                "Combined files should be handled before calling pixel_to_video."
            )

        video_rect = tuple(float(v) for v in self.alignments.get_video_rect())
        image_width, image_height = image.size
        return self._pixel_to_video(
            x_pixel, y_pixel, video_rect, image_width, image_height
        )

    def get_scanner_coordinates_at_pixel(
        self,
        x_pixel: float,
        y_pixel: float,
    ) -> list[dict[str, Any]] | list[Any]:
        """Return scanner coordinates for each 2D alignment at an image pixel.

        Parameters
        ----------
        x_pixel, y_pixel
            Image pixel coordinates.

        Returns
        -------
        list[dict[str, Any]] | list[Any]
            For a normal file, returns a list of dictionaries, one per 2D
            alignment, containing:
            - ``alignment_index``
            - ``video_xy``
            - ``scanner_xy``

            For a combined file, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_scanner_coordinates_at_pixel",
                x_pixel,
                y_pixel,
            )

        image = self.get_image()
        if isinstance(image, list):
            raise ValueError(
                "Combined files should be handled before calling "
                "get_scanner_coordinates_at_pixel."
            )

        video_rect = tuple(float(v) for v in self.alignments.get_video_rect())
        image_width, image_height = image.size
        video_x, video_y = self._pixel_to_video(
            x_pixel, y_pixel, video_rect, image_width, image_height
        )

        results: list[dict[str, Any]] = []
        for i, alignment2d in enumerate(self.alignments.alignments_2d):
            scanner_xy = alignment2d.video_to_scanner(video_x, video_y)
            try:
                scanner_x, scanner_y = scanner_xy
            except Exception as e:
                raise ValueError(
                    f"Alignment2D.video_to_scanner did not return a 2-tuple for "
                    f"alignment index {i}: {scanner_xy!r}"
                ) from e

            results.append(
                {
                    "alignment_index": i,
                    "video_xy": (float(video_x), float(video_y)),
                    "scanner_xy": (float(scanner_x), float(scanner_y)),
                }
            )

        return results

    def get_beam_rays_at_pixel(
        self,
        x_pixel: float,
        y_pixel: float,
        *,
        distance0: float = 0.0,
        distance1: float = 1.0,
    ) -> list[dict[str, Any]] | list[Any]:
        """Return beam origin and unit direction for each scan head at a pixel.

        Parameters
        ----------
        x_pixel, y_pixel
            Image pixel coordinates.
        distance0, distance1
            Distances passed to ``Alignment3D.scanner_to_coord_3d`` to construct
            two points along the beam. ``distance0`` defines the origin point,
            and the vector from that point to the point at ``distance1`` is
            normalized to produce the unit beam direction.

        Returns
        -------
        list[dict[str, Any]] | list[Any]
            For a normal file, returns a list of dictionaries, one per matched
            2D/3D alignment pair, containing:
            - ``alignment_index``
            - ``video_xy``
            - ``scanner_xy``
            - ``origin``
            - ``direction``

            For a combined file, returns a list of per-contained-file results.

        Raises
        ------
        ValueError
            If the number of 2D and 3D alignments does not match, or if a zero
            direction vector is encountered.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_beam_rays_at_pixel",
                x_pixel,
                y_pixel,
                distance0=distance0,
                distance1=distance1,
            )

        alignments_2d = self.alignments.alignments_2d
        alignments_3d = self.alignments.alignments_3d

        if len(alignments_2d) != len(alignments_3d):
            raise ValueError(
                "The number of 2D alignments does not match the number of 3D "
                "alignments, so scan-head pairing by index is ambiguous."
            )

        scanner_results = self.get_scanner_coordinates_at_pixel(x_pixel, y_pixel)
        if not isinstance(scanner_results, list):
            raise ValueError(
                "Unexpected result from get_scanner_coordinates_at_pixel."
            )

        results: list[dict[str, Any]] = []

        for result in scanner_results:
            i = int(result["alignment_index"])
            scanner_x, scanner_y = result["scanner_xy"]
            alignment3d = alignments_3d[i]

            p0 = alignment3d.scanner_to_coord_3d(scanner_x, scanner_y, distance0)
            p1 = alignment3d.scanner_to_coord_3d(scanner_x, scanner_y, distance1)

            try:
                p0_arr = np.asarray(p0, dtype=float).reshape(3)
                p1_arr = np.asarray(p1, dtype=float).reshape(3)
            except Exception as exc:
                raise ValueError(
                    f"Alignment3D.scanner_to_coord_3d returned unexpected values "
                    f"for alignment index {i}: p0={p0!r}, p1={p1!r}"
                ) from exc

            direction = p1_arr - p0_arr
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                raise ValueError(
                    f"Zero-length beam direction encountered for alignment index {i}."
                )
            direction_unit = direction / norm

            results.append(
                {
                    "alignment_index": i,
                    "video_xy": result["video_xy"],
                    "scanner_xy": (float(scanner_x), float(scanner_y)),
                    "origin": p0_arr,
                    "direction": direction_unit,
                }
            )

        return results

    def get_beam_intersection_at_pixel(
        self,
        x_pixel: float,
        y_pixel: float,
        *,
        distance0: float = 0.0,
        distance1: float = 1.0,
    ) -> dict[str, Any] | list[Any]:
        """Compute the least-squares beam intersection point at an image pixel.

        Parameters
        ----------
        x_pixel, y_pixel
            Image pixel coordinates.
        distance0, distance1
            Distances passed to ``Alignment3D.scanner_to_coord_3d`` to construct
            two points on each beam.

        Returns
        -------
        dict[str, Any] | list[Any]
            For a normal file, returns a dictionary containing:
            - ``pixel_xy``
            - ``video_xy``
            - ``intersection``
            - ``distances``
            - ``rms_distance``
            - ``mean_distance``
            - ``max_distance``
            - ``beam_rays``

            For a combined file, returns a list of per-contained-file results.

        Raises
        ------
        ValueError
            If fewer than two beam rays are available.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_beam_intersection_at_pixel",
                x_pixel,
                y_pixel,
                distance0=distance0,
                distance1=distance1,
            )

        beam_rays = self.get_beam_rays_at_pixel(
            x_pixel,
            y_pixel,
            distance0=distance0,
            distance1=distance1,
        )

        if not isinstance(beam_rays, list):
            raise ValueError("Unexpected result from get_beam_rays_at_pixel.")

        if len(beam_rays) < 2:
            raise ValueError(
                "At least two beam rays are required to compute an intersection."
            )

        try:
            video_xy = beam_rays[0]["video_xy"]
        except Exception:
            video_xy = self.pixel_to_video(x_pixel, y_pixel)

        P0 = np.asarray([ray["origin"] for ray in beam_rays], dtype=float)
        directions = np.asarray([ray["direction"] for ray in beam_rays], dtype=float)
        P1 = P0 + directions

        intersection = np.asarray(
            intersection_point_multiple_lines(P0, P1),
            dtype=float,
        ).reshape(3)

        distances = np.asarray(
            [
                distance_point_line(
                    intersection[np.newaxis, :],
                    ray["origin"][np.newaxis, :],
                    ray["direction"][np.newaxis, :],
                )[0]
                for ray in beam_rays
            ],
            dtype=float,
        )

        return {
            "pixel_xy": (float(x_pixel), float(y_pixel)),
            "video_xy": tuple(float(v) for v in video_xy),
            "intersection": intersection,
            "distances": distances,
            "rms_distance": float(np.sqrt(np.mean(distances**2))),
            "mean_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
            "beam_rays": beam_rays,
        }

    def _get_measpoint_pixel_coordinates(self) -> np.ndarray:
        """Return measurement-point pixel coordinates as an ``(n, 2)`` array."""
        image = self.get_image()
        if isinstance(image, list):
            raise ValueError(
                "Combined files should be handled before calling "
                "_get_measpoint_pixel_coordinates."
            )

        video_rect = tuple(float(v) for v in self.measurement_points.get_video_rect())
        image_width, image_height = image.size

        pixel_points: list[tuple[float, float]] = []
        for pt in self.measurement_points:
            try:
                x_video, y_video = pt.video_xy()
            except Exception:
                continue
            pixel_points.append(
                self._video_to_pixel(
                    float(x_video),
                    float(y_video),
                    video_rect,
                    image_width,
                    image_height,
                )
            )

        if not pixel_points:
            return np.empty((0, 2), dtype=float)

        return np.asarray(pixel_points, dtype=float)

    @staticmethod
    def _make_convex_hull_path(
        pixel_points: np.ndarray,
        *,
        buffer_pixels: float = 0.0,
    ) -> Any:
        """Create a matplotlib Path representing the convex hull of points.

        Parameters
        ----------
        pixel_points
            Array of shape ``(n, 2)`` of pixel coordinates.
        buffer_pixels
            Optional outward buffer distance in pixels. If zero, the raw convex
            hull polygon is used.

        Returns
        -------
        matplotlib.path.Path | None
            Path for the hull polygon, or ``None`` if fewer than three points
            are available.
        """
        if pixel_points.shape[0] < 3:
            return None

        from scipy.spatial import ConvexHull
        from matplotlib.path import Path

        hull = ConvexHull(pixel_points)
        hull_points = pixel_points[hull.vertices]

        if buffer_pixels > 0.0:
            centroid = hull_points.mean(axis=0)
            shifted = hull_points - centroid
            norms = np.linalg.norm(shifted, axis=1)
            norms[norms == 0.0] = 1.0
            hull_points = centroid + shifted * (
                (norms + float(buffer_pixels)) / norms
            )[:, np.newaxis]

        return Path(hull_points)

    @staticmethod
    def _grid_points_from_mesh(
        xx: np.ndarray,
        yy: np.ndarray,
    ) -> np.ndarray:
        """Convert meshgrid arrays into an ``(n, 2)`` array of point coordinates."""
        return np.column_stack((xx.ravel(), yy.ravel()))

    def compute_alignment_error_map(
        self,
        *,
        pixel_step: int = 10,
        region_mode: str = "full",
        hull_buffer_pixels: float = 0.0,
        scale_region_mode: str = "measpoint_hull",
        metric: str = "rms",
        distance0: float = 0.0,
        distance1: float = 1.0,
    ) -> dict[str, Any] | list[Any]:
        """Compute a sampled alignment-error field over the image.

        Parameters
        ----------
        pixel_step
            Sampling stride in pixels in both x and y directions.
        region_mode
            Region over which to compute the field. Supported values are:
            - ``"full"``
            - ``"measpoint_hull"``
        hull_buffer_pixels
            Optional outward buffer applied to the measurement-point convex hull.
        scale_region_mode
            Region used to determine colormap scaling values. Supported values:
            - ``"full"``
            - ``"measpoint_hull"``
        metric
            Error metric extracted from ``get_beam_intersection_at_pixel``.
            Supported values:
            - ``"rms"``
            - ``"mean"``
            - ``"max"``
        distance0, distance1
            Distances used to construct beam rays.

        Returns
        -------
        dict[str, Any] | list[Any]
            For a normal file, returns a dictionary containing:
            - ``x_grid``
            - ``y_grid``
            - ``error``
            - ``compute_mask``
            - ``scale_mask``
            - ``metric``
            - ``region_mode``
            - ``scale_region_mode``
            - ``pixel_step``
            - ``hull_points``
            - ``vmin``
            - ``vmax``

            For a combined file, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "compute_alignment_error_map",
                pixel_step=pixel_step,
                region_mode=region_mode,
                hull_buffer_pixels=hull_buffer_pixels,
                scale_region_mode=scale_region_mode,
                metric=metric,
                distance0=distance0,
                distance1=distance1,
            )

        metric_l = metric.lower()
        if metric_l not in {"rms", "mean", "max"}:
            raise ValueError("`metric` must be one of 'rms', 'mean', or 'max'.")

        region_mode_l = region_mode.lower()
        if region_mode_l not in {"full", "measpoint_hull"}:
            raise ValueError("`region_mode` must be 'full' or 'measpoint_hull'.")

        scale_region_mode_l = scale_region_mode.lower()
        if scale_region_mode_l not in {"full", "measpoint_hull"}:
            raise ValueError(
                "`scale_region_mode` must be 'full' or 'measpoint_hull'."
            )

        if pixel_step <= 0:
            raise ValueError("`pixel_step` must be a positive integer.")

        image = self.get_image()
        if isinstance(image, list):
            raise ValueError(
                "Combined files should be handled before calling "
                "compute_alignment_error_map."
            )

        image_width, image_height = image.size

        x_vals = np.arange(0, image_width, pixel_step, dtype=float)
        y_vals = np.arange(0, image_height, pixel_step, dtype=float)
        if len(x_vals) == 0 or x_vals[-1] != image_width - 1:
            x_vals = np.append(x_vals, image_width - 1)
        if len(y_vals) == 0 or y_vals[-1] != image_height - 1:
            y_vals = np.append(y_vals, image_height - 1)

        xx, yy = np.meshgrid(x_vals, y_vals)
        error = np.full(xx.shape, np.nan, dtype=float)

        measpoint_pixels = self._get_measpoint_pixel_coordinates()
        hull_path = self._make_convex_hull_path(
            measpoint_pixels,
            buffer_pixels=hull_buffer_pixels,
        )

        if hull_path is not None:
            grid_points = self._grid_points_from_mesh(xx, yy)
            hull_mask_flat = hull_path.contains_points(grid_points)
            hull_mask = hull_mask_flat.reshape(xx.shape)
        else:
            hull_mask = np.zeros(xx.shape, dtype=bool)

        if region_mode_l == "full":
            compute_mask = np.ones(xx.shape, dtype=bool)
        else:
            if hull_path is None:
                compute_mask = np.ones(xx.shape, dtype=bool)
            else:
                compute_mask = hull_mask

        for iy in range(xx.shape[0]):
            for ix in range(xx.shape[1]):
                if not compute_mask[iy, ix]:
                    continue

                x_pixel = float(xx[iy, ix])
                y_pixel = float(yy[iy, ix])

                try:
                    result = self.get_beam_intersection_at_pixel(
                        x_pixel,
                        y_pixel,
                        distance0=distance0,
                        distance1=distance1,
                    )
                    if metric_l == "rms":
                        value = result["rms_distance"]
                    elif metric_l == "mean":
                        value = result["mean_distance"]
                    else:
                        value = result["max_distance"]
                    error[iy, ix] = float(value)
                except Exception:
                    error[iy, ix] = np.nan

        if scale_region_mode_l == "full":
            scale_mask = np.isfinite(error)
        else:
            if hull_path is None:
                scale_mask = np.isfinite(error)
            else:
                scale_mask = hull_mask & np.isfinite(error)

        if not np.any(scale_mask):
            vmin = np.nan
            vmax = np.nan
        else:
            scale_values = error[scale_mask]
            vmin = float(np.nanmin(scale_values))
            vmax = float(np.nanmax(scale_values))

        hull_points = None
        if hull_path is not None:
            hull_points = np.asarray(hull_path.vertices, dtype=float)

        return {
            "x_grid": xx,
            "y_grid": yy,
            "error": error,
            "compute_mask": compute_mask,
            "scale_mask": scale_mask,
            "metric": metric_l,
            "region_mode": region_mode_l,
            "scale_region_mode": scale_region_mode_l,
            "pixel_step": int(pixel_step),
            "hull_points": hull_points,
            "vmin": vmin,
            "vmax": vmax,
        }

    def get_alignment_error_image(
        self,
        *,
        pixel_step: int = 10,
        region_mode: str = "full",
        hull_buffer_pixels: float = 0.0,
        scale_region_mode: str = "measpoint_hull",
        metric: str = "max",
        distance0: float = 0.0,
        distance1: float = 1.0,
        cmap: str = "viridis",
        alpha: float = 0.6,
        show_measurement_points: bool = False,
        show_hull: bool = False,
        ax: Any = None,
        title: str | None = None,
    ) -> Any:
        """Overlay a sampled alignment-error colormap on the stored image.

        Parameters
        ----------
        pixel_step
            Sampling stride in pixels.
        region_mode
            Region over which to compute the field:
            - ``"full"``
            - ``"measpoint_hull"``
        hull_buffer_pixels
            Optional outward buffer applied to the measurement-point convex hull.
        scale_region_mode
            Region used to set colormap scaling:
            - ``"full"``
            - ``"measpoint_hull"``
        metric
            Error metric to display:
            - ``"rms"``
            - ``"mean"``
            - ``"max"``
        distance0, distance1
            Distances used to construct beam rays.
        cmap
            Matplotlib colormap name.
        alpha
            Overlay transparency.
        show_measurement_points
            Whether to overlay measurement-point positions.
        show_hull
            Whether to draw the convex hull polygon used for hull-based masking.
        ax
            Existing matplotlib axes to draw onto.
        title
            Plot title. If omitted, a default title is generated.

        Returns
        -------
        Any
            For a normal file, returns ``(fig, ax, result_dict)``.
            For a combined file, returns a list of per-contained-file results.
        """
        contained = self._get_contained_files_or_none()
        if contained is not None:
            return self._delegate_image_method_to_contained_files(
                "get_alignment_error_image",
                pixel_step=pixel_step,
                region_mode=region_mode,
                hull_buffer_pixels=hull_buffer_pixels,
                scale_region_mode=scale_region_mode,
                metric=metric,
                distance0=distance0,
                distance1=distance1,
                cmap=cmap,
                alpha=alpha,
                show_measurement_points=show_measurement_points,
                show_hull=show_hull,
                ax=ax,
                title=title,
            )

        image = self.get_image()
        if isinstance(image, list):
            raise ValueError(
                "Combined files should be handled before calling "
                "get_alignment_error_image."
            )

        result = self.compute_alignment_error_map(
            pixel_step=pixel_step,
            region_mode=region_mode,
            hull_buffer_pixels=hull_buffer_pixels,
            scale_region_mode=scale_region_mode,
            metric=metric,
            distance0=distance0,
            distance1=distance1,
        )
        if isinstance(result, list):
            raise ValueError("Unexpected combined-file result during plotting.")

        xx = result["x_grid"]
        yy = result["y_grid"]
        error = result["error"]
        vmin = result["vmin"]
        vmax = result["vmax"]
        hull_points = result["hull_points"]

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        else:
            fig = ax.figure

        ax.imshow(image)

        mesh = ax.pcolormesh(
            xx,
            yy,
            error,
            shading="nearest",
            cmap=cmap,
            alpha=alpha,
            vmin=None if np.isnan(vmin) else vmin,
            vmax=None if np.isnan(vmax) else vmax,
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(f"{result['metric']} beam intersection error")

        if show_measurement_points:
            measpoint_pixels = self._get_measpoint_pixel_coordinates()
            if measpoint_pixels.size > 0:
                ax.scatter(
                    measpoint_pixels[:, 0],
                    measpoint_pixels[:, 1],
                    c="white",
                    edgecolors="black",
                    s=20,
                    marker="o",
                    label="Measurement Points",
                )

        if show_hull and hull_points is not None and len(hull_points) > 0:
            closed_hull = np.vstack((hull_points, hull_points[0]))
            ax.plot(
                closed_hull[:, 0],
                closed_hull[:, 1],
                color="white",
                linestyle="--",
                linewidth=1.5,
                label="Measurement-Point Hull",
            )

        image_width, image_height = image.size
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        if title is None:
            title = (
                f"Alignment Error Map ({result['metric']}, step={result['pixel_step']}, "
                f"region={result['region_mode']}, scale={result['scale_region_mode']})"
            )
        ax.set_title(title)

        if show_measurement_points or (show_hull and hull_points is not None):
            ax.legend()

        if created_fig:
            fig.tight_layout()

        return fig, ax, result