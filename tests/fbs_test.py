"""
Created Wed October 1 2025

@author: spcarte
"""

import numpy as np
import sdynpy as sdpy
import pytest

# Defining global parameters for the test systems
freq = np.arange(601)*0.1
m1 = 1 
m2 = 1 
m3 = 1
m4 = 1
m5 = 1
m6 = 1 
m7 = 1 

k1 = 10000
k2 = 20000
k34 = 30000
k45 = 40000
k6 = 60000
k7 = 70000

c1 = 1
c2 = 2
c34 = 3
c45 = 4
c6 = 6
c7 = 7

@pytest.fixture
def system_a_frfs():
    Ma = np.array([[m1, 0],
                   [0,  m2]])
    
    Ka = np.array([[k1+k2, -k2],
                   [-k2,    k2]])
    
    Ca = np.array([[c1+c2, -c2],
                   [-c2,    c2]])
    
    dof_a = sdpy.coordinate_array(node=[1,2], direction=1)
    system_a = sdpy.System(dof_a, Ma, Ka, Ca)
    return system_a.frequency_response(frequencies=freq)

@pytest.fixture
def system_b_frfs():
    Mb = np.array([[m3, 0,  0],
                   [0,  m4, 0],
                   [0,  0,  m5]])

    Kb = np.array([[k34+10, -k34,     0], 
                   [-k34, k34+k45, -k45],
                   [0,   -k45,      k45]])

    Cb = np.array([[c34, -c34,     0],
                   [-c34, c34+c45, -c45],
                   [0,   -c45,      c45]])

    dof_b = sdpy.coordinate_array(node=[3,4,5], direction=1)
    system_b = sdpy.System(dof_b, Mb, Kb, Cb)
    return system_b.frequency_response(frequencies=freq)

@pytest.fixture
def system_c_frfs():
    Mc = np.array([[m6, 0],
                   [0,  m7]])

    Kc = np.array([[k6, -k6],
                   [-k6, k6+k7]])

    Cc = np.array([[c6, -c6],
                   [-c6, c6+c7]])

    dof_c = sdpy.coordinate_array(node=[6,7], direction=1)
    system_c = sdpy.System(dof_c, Mc, Kc, Cc)
    return system_c.frequency_response(frequencies=freq)

@pytest.fixture
def system_ab_frfs():
    Mab = np.array([[m1, 0,     0,  0],
                    [0,  m2+m3, 0,  0],
                    [0,  0,     m4, 0],
                    [0,  0,     0,  m5]])

    Kab = np.array([[k1+k2, -k2,         0,        0],
                    [-k2,    k2+k34+10, -k34,      0],
                    [0,     -k34,        k34+k45, -k45],
                    [0,      0,         -k45,      k45]])

    Cab = np.array([[c1+c2, -c2,      0,        0],
                    [-c2,    c2+c34, -c34,      0],
                    [0,     -c34,     c34+c45, -c45],
                    [0,      0,      -c45,      c45]])

    dof_ab = sdpy.coordinate_array(node=[1,2,4,5], direction=1)
    system_ab = sdpy.System(dof_ab, Mab, Kab, Cab)
    return system_ab.frequency_response(frequencies=freq)

@pytest.fixture
def system_ac_frfs():
    Mac = np.array([[m1, 0,     0],
                    [0,  m2+m6, 0],
                    [0,  0,     m7]])

    Kac = np.array([[k1+k2, -k2,     0],
                    [-k2,    k2+k6, -k6],
                    [0,     -k6,     k6+k7]])

    Cac = np.array([[c1+c2, -c2,     0],
                    [-c2,    c2+c6, -c6],
                    [0,     -c6,     c6+c7]])

    dof_ac = sdpy.coordinate_array(node=[1,2,7], direction=1)
    system_ac = sdpy.System(dof_ac, Mac, Kac, Cac)
    return system_ac.frequency_response(frequencies=freq)

@pytest.fixture
def system_abc_frfs():
    Mabc = np.array([[m1, 0,     0,  0,     0],
                     [0,  m2+m3, 0,  0,     0],
                     [0,  0,     m4, 0,     0],
                     [0,  0,     0,  m5+m6, 0], 
                     [0,  0,     0,  0,     m7]])

    Kabc = np.array([[k1+k2, -k2,         0,        0,       0],
                     [-k2,    k2+k34+10, -k34,      0,       0],
                     [0,     -k34,        k34+k45, -k45,     0],
                     [0,      0,         -k45,      k45+k6, -k6],
                     [0,      0,          0,       -k6,      k6+k7]])

    Cabc = np.array([[c1+c2, -c2,      0,        0,       0],
                     [-c2,    c2+c34, -c34,      0,       0],
                     [0,     -c34,     c34+c45, -c45,     0],
                     [0,      0,      -c45,      c45+c6, -c6],
                     [0,      0,       0,       -c6,      c6+c7]])

    dof_abc = sdpy.coordinate_array(node=[1,2,4,6,7], direction=1)
    system_abc = sdpy.System(dof_abc, Mabc, Kabc, Cabc)
    return system_abc.frequency_response(frequencies=freq)

def test_simple_example(system_a_frfs, system_c_frfs, system_ac_frfs):
    """
    Tests that the FBS process works to couple two 2x2 system. 
    """
    ac_block_diagonal = sdpy.TransferFunctionArray.block_diagonal_frf((system_a_frfs, system_c_frfs))
    ac_coupling_coordinate = sdpy.coordinate_array(string_array=['2X+', '6X+'])[np.newaxis,...]
    ac_substructure_frfs = ac_block_diagonal.substructure_by_coordinate(ac_coupling_coordinate)

    # Test that the block diagonal function puts things in the right place
    assert np.all(ac_block_diagonal[:2,:2].ordinate == system_a_frfs.ordinate), 'Unexpected values for the upper left part of the block diagonal FRFs'
    assert np.all(ac_block_diagonal[2:,2:].ordinate == system_c_frfs.ordinate), 'Unexpected values for the lower right part of the block diagonal FRFs'
    assert np.all(ac_block_diagonal[:2,2:].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF'
    assert np.all(ac_block_diagonal[2:,:2].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF'

    # Test that the substructuring works
    assert np.allclose(system_ac_frfs.ordinate, ac_substructure_frfs[system_ac_frfs.coordinate].ordinate), 'the substructure coupled FRFs do not match the truth FRFs'
    assert np.allclose(ac_substructure_frfs[1,:].ordinate, ac_substructure_frfs[2,:].ordinate), 'the substructuring did not result in duplicate dofs'
    assert np.allclose(ac_substructure_frfs[:,1].ordinate, ac_substructure_frfs[:,2].ordinate), 'the substructuring did not result in duplicate dofs'

def test_three_system_fbs(system_a_frfs, system_b_frfs, system_c_frfs, system_abc_frfs):
    """
    Test that the FBS process works to couple three systems. Two systems are 2x2 
    and the third is 3x3.
    """
    abc_block_diagonal = sdpy.TransferFunctionArray.block_diagonal_frf((system_a_frfs, system_b_frfs, system_c_frfs))
    abc_coupling_coordinate = sdpy.coordinate_array(string_array=[['2X+', '3X+'],
                                                                  ['5X+', '6X+']])
    abc_substructure_frfs = abc_block_diagonal.substructure_by_coordinate(abc_coupling_coordinate)

    # Test that the block diagonal function puts things in the right place
    assert np.all(abc_block_diagonal[:2,:2].ordinate == system_a_frfs.ordinate), 'Unexpected values for the upper left part of the block diagonal FRFs'
    assert np.all(abc_block_diagonal[2:5,2:5].ordinate == system_b_frfs.ordinate), 'Unexpected values for the middle part of the block diagonal FRFs'
    assert np.all(abc_block_diagonal[-2:,-2:].ordinate == system_c_frfs.ordinate), 'Unexpected values for the lower right part of the block diagonal FRFs'
    assert np.all(abc_block_diagonal[2:,:2].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF'
    assert np.all(abc_block_diagonal[:2,2:].ordinate == 0 ), 'There are non-zero values in the off-diagonal of the block diagonal FRF'
    assert np.all(abc_block_diagonal[-2:,:-2].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF'
    assert np.all(abc_block_diagonal[:-2:,-2:].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF' 

    # Test that the substructuring works
    assert np.allclose(system_abc_frfs.ordinate, abc_substructure_frfs[system_abc_frfs.coordinate].ordinate), 'the substructure coupled FRFs do not match the truth FRFs'
    assert np.allclose(abc_substructure_frfs[:, 1].ordinate, abc_substructure_frfs[:, 2].ordinate), 'the substructuring did not result in duplicate dofs'
    assert np.allclose(abc_substructure_frfs[1, :].ordinate, abc_substructure_frfs[2, :].ordinate), 'the substructuring did not result in duplicate dofs'
    assert np.allclose(abc_substructure_frfs[-3, :].ordinate, abc_substructure_frfs[-2, :].ordinate), 'the substructuring did not result in duplicate dofs'
    assert np.allclose(abc_substructure_frfs[:, -3].ordinate, abc_substructure_frfs[:, -2].ordinate), 'the substructuring did not result in duplicate dofs'

def test_rectangular_frf_fbs(system_a_frfs, system_b_frfs, system_ab_frfs):
    """
    Tests that the FBS coupling process works with rectangular FRF matrices.
    """
    ab_couple_block_diagonal = sdpy.TransferFunctionArray.block_diagonal_frf((system_a_frfs[:,1], system_b_frfs[:,0]))
    ab_coupling_coordinate = sdpy.coordinate_array(string_array=[['2X+', '3X+']])
    ab_substructure_frfs = ab_couple_block_diagonal.substructure_by_coordinate(ab_coupling_coordinate)
    
    # Test that the block diagonal function puts things in the right place
    assert np.all(ab_couple_block_diagonal[:2,0].ordinate == system_a_frfs[:,1].ordinate)
    assert np.all(ab_couple_block_diagonal[-3:,-1].ordinate == system_b_frfs[:,0].ordinate)
    assert np.all(ab_couple_block_diagonal[2:,0].ordinate == 0)
    assert np.all(ab_couple_block_diagonal[:-3,-1].ordinate == 0)

    # Test that the substructuring works
    assert np.allclose(system_ab_frfs[:,1].ordinate, ab_substructure_frfs[system_ab_frfs[:,1].coordinate].ordinate)
    assert np.allclose(ab_substructure_frfs[:,0].ordinate, ab_substructure_frfs[:,1].ordinate)
    assert np.allclose(ab_substructure_frfs[1,:].ordinate, ab_substructure_frfs[2,:].ordinate)

def test_two_system_decoupling(system_abc_frfs, system_ab_frfs, system_c_frfs):
    """
    Tests that the FBS process works for decoupling one system from another.
    """
    ab_decouple_block_diagonal = sdpy.TransferFunctionArray.block_diagonal_frf((system_abc_frfs, -system_c_frfs), coordinate_node_offset=10)
    ab_decoupling_coordinate = sdpy.coordinate_array(string_array=[['16X+', '26X+'],
                                                                   ['17X+', '27X+']])
    ab_decoupled_frfs = ab_decouple_block_diagonal.substructure_by_coordinate(ab_decoupling_coordinate)

    # Test that the block diagonal function puts things in the right place
    assert np.all(ab_decouple_block_diagonal[:5,:5].ordinate == system_abc_frfs.ordinate), 'Unexpected values for the upper left part of the block diagonal FRFs'
    assert np.all(ab_decouple_block_diagonal[-2:,-2:].ordinate == -system_c_frfs.ordinate), 'Unexpected values for the lower right part of the block diagonal FRFs'
    assert np.all(ab_decouple_block_diagonal[5:,:5].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF' 
    assert np.all(ab_decouple_block_diagonal[:5,5:].ordinate == 0), 'There are non-zero values in the off-diagonal of the block diagonal FRF' 

    # Test that the substructuring works, an allclose comparison isn't used here because 
    # the decoupling process is extremely sensitive to numerical precision and will not 
    # pass an allclose test 
    assert np.max(np.abs(((ab_decoupled_frfs[:4,:4] - system_ab_frfs)/system_ab_frfs).ordinate)) < 0.05, 'The substructure decoupling has too much error'