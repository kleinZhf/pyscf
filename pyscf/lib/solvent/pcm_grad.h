/*
 * Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Header file for PCM gradient calculations
 */

#ifndef PCM_GRAD_H
#define PCM_GRAD_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * C implementation of get_dF_dA function for PCM gradient calculations
 * Based on J. Chem. Phys. 133, 244111 (2010), Appendix C
 * 
 * Input parameters:
 * - atom_coords: atomic coordinates (natom x 3)
 * - grid_coords: grid coordinates (ngrids x 3) 
 * - switch_fun: switch function values (ngrids)
 * - area: area values (ngrids)
 * - R_in_J: inner radius parameter (natom)
 * - R_sw_J: switching radius parameter (natom)
 * - gslice_by_atom: grid slicing information for atoms (natom x 2)
 * - natom: number of atoms
 * - ngrids: number of grid points
 * 
 * Output parameters:
 * - dF: grid derivatives (ngrids x natom x 3)
 * - dA: area derivatives (ngrids x natom x 3)
 * 
 * Returns 0 on success, non-zero on error
 */
int PCM_get_dF_dA(const double *atom_coords, const double *grid_coords,
                  const double *switch_fun, const double *area,
                  const double *R_in_J, const double *R_sw_J,
                  const int *gslice_by_atom,
                  int natom, int ngrids,
                  double *dF, double *dA);

/*
 * Wrapper function that can be called from Python via ctypes
 * All arrays are assumed to be contiguous and in C order
 */
int PCM_get_dF_dA_wrapper(double *atom_coords, double *grid_coords,
                         double *switch_fun, double *area,
                         double *R_in_J, double *R_sw_J, 
                         int *gslice_by_atom,
                         int natom, int ngrids,
                         double *dF, double *dA);

#ifdef __cplusplus
}
#endif

#endif /* PCM_GRAD_H */