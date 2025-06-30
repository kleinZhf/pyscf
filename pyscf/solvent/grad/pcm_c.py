#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: AI Assistant
#

"""
Python bindings for the C++ implementation of get_dF_dA function
"""

import numpy
import ctypes
from pyscf import lib

# Load the solvent library
libsolvent = lib.load_library('libsolvent')

def get_dF_dA_c(surface):
    """
    C implementation of get_dF_dA function
    
    J. Chem. Phys. 133, 244111 (2010), Appendix C
    
    Parameters:
    -----------
    surface : dict
        Surface dictionary containing:
        - atom_coords: Atomic coordinates (natom x 3)
        - grid_coords: Grid coordinates (ngrids x 3) 
        - switch_fun: Switch function values (ngrids)
        - area: Area values (ngrids)
        - R_in_J: Inner radius parameter (natom)
        - R_sw_J: Switching radius parameter (natom)
        - gslice_by_atom: Grid slicing information for atoms (natom x 2)
    
    Returns:
    --------
    dF : ndarray
        Grid derivatives (ngrids x natom x 3)
    dA : ndarray
        Area derivatives (ngrids x natom x 3)
    """
    
    # Extract surface data
    atom_coords = numpy.asarray(surface['atom_coords'], dtype=numpy.float64, order='C')
    grid_coords = numpy.asarray(surface['grid_coords'], dtype=numpy.float64, order='C')
    switch_fun = numpy.asarray(surface['switch_fun'], dtype=numpy.float64, order='C')
    area = numpy.asarray(surface['area'], dtype=numpy.float64, order='C')
    R_in_J = numpy.asarray(surface['R_in_J'], dtype=numpy.float64, order='C')
    R_sw_J = numpy.asarray(surface['R_sw_J'], dtype=numpy.float64, order='C')
    gslice_by_atom = numpy.asarray(surface['gslice_by_atom'], dtype=numpy.int32, order='C')
    
    natom = atom_coords.shape[0]
    ngrids = grid_coords.shape[0]
    
    # Input validation
    assert atom_coords.shape == (natom, 3), f"atom_coords shape mismatch: {atom_coords.shape} != {(natom, 3)}"
    assert grid_coords.shape == (ngrids, 3), f"grid_coords shape mismatch: {grid_coords.shape} != {(ngrids, 3)}"
    assert switch_fun.shape == (ngrids,), f"switch_fun shape mismatch: {switch_fun.shape} != {(ngrids,)}"
    assert area.shape == (ngrids,), f"area shape mismatch: {area.shape} != {(ngrids,)}"
    assert R_in_J.shape == (natom,), f"R_in_J shape mismatch: {R_in_J.shape} != {(natom,)}"
    assert R_sw_J.shape == (natom,), f"R_sw_J shape mismatch: {R_sw_J.shape} != {(natom,)}"
    assert gslice_by_atom.shape == (natom, 2), f"gslice_by_atom shape mismatch: {gslice_by_atom.shape} != {(natom, 2)}"
    
    # Allocate output arrays
    dF = numpy.zeros((ngrids, natom, 3), dtype=numpy.float64, order='C')
    dA = numpy.zeros((ngrids, natom, 3), dtype=numpy.float64, order='C')
    
    # Set up the C function signature
    fn = libsolvent.PCM_get_dF_dA_wrapper
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # atom_coords
        ctypes.POINTER(ctypes.c_double),  # grid_coords
        ctypes.POINTER(ctypes.c_double),  # switch_fun
        ctypes.POINTER(ctypes.c_double),  # area
        ctypes.POINTER(ctypes.c_double),  # R_in_J
        ctypes.POINTER(ctypes.c_double),  # R_sw_J
        ctypes.POINTER(ctypes.c_int),     # gslice_by_atom
        ctypes.c_int,                     # natom
        ctypes.c_int,                     # ngrids
        ctypes.POINTER(ctypes.c_double),  # dF
        ctypes.POINTER(ctypes.c_double),  # dA
    ]
    
    # Convert numpy arrays to ctypes pointers
    atom_coords_ptr = atom_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    grid_coords_ptr = grid_coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    switch_fun_ptr = switch_fun.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    area_ptr = area.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    R_in_J_ptr = R_in_J.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    R_sw_J_ptr = R_sw_J.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    gslice_by_atom_ptr = gslice_by_atom.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dF_ptr = dF.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    dA_ptr = dA.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # Call the C function
    ret_code = fn(
        atom_coords_ptr, grid_coords_ptr, switch_fun_ptr, area_ptr,
        R_in_J_ptr, R_sw_J_ptr, gslice_by_atom_ptr,
        natom, ngrids, dF_ptr, dA_ptr
    )
    
    if ret_code != 0:
        raise RuntimeError(f"PCM_get_dF_dA_wrapper returned error code: {ret_code}")
    
    return dF, dA