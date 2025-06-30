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
 * C implementation of get_dF_dA function for PCM gradient calculations
 * Based on J. Chem. Phys. 133, 244111 (2010), Appendix C
 * 
 * Author: AI Assistant
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Switch function h(x) from Eq. 3.19 in J. Chem. Phys. 133, 244111 (2010)
 * h(x) = x^3 * (10 - 15*x + 6*x^2) for 0 <= x <= 1
 * h(x) = 0 for x < 0
 * h(x) = 1 for x > 1
 */
static double switch_h(double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double x2 = x * x;
    double x3 = x2 * x;
    return x3 * (10.0 - 15.0*x + 6.0*x2);
}

/*
 * First derivative of switch function h(x)
 * dh/dx = 30*x^2 - 60*x^3 + 30*x^4 for 0 <= x <= 1
 * dh/dx = 0 for x < 0 or x > 1
 */
static double grad_switch_h(double x) {
    if (x <= 0.0 || x >= 1.0) return 0.0;
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x2 * x2;
    return 30.0*x2 - 60.0*x3 + 30.0*x4;
}

/*
 * Calculate Euclidean distance between two 3D points
 */
static double distance_3d(const double *p1, const double *p2) {
    double dx = p1[0] - p2[0];
    double dy = p1[1] - p2[1];
    double dz = p1[2] - p2[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/*
 * C implementation of get_dF_dA function
 * 
 * Calculates derivatives according to J. Chem. Phys. 133, 244111 (2010), Appendix C
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
                  double *dF, double *dA) {
    
    // Input validation
    if (!atom_coords || !grid_coords || !switch_fun || !area || 
        !R_in_J || !R_sw_J || !gslice_by_atom || !dF || !dA) {
        return -1; // Null pointer error
    }
    
    if (natom <= 0 || ngrids <= 0) {
        return -2; // Invalid dimensions
    }
    
    // Initialize output arrays to zero
    memset(dF, 0, ngrids * natom * 3 * sizeof(double));
    memset(dA, 0, ngrids * natom * 3 * sizeof(double));
    
    // Main computation loop over atoms
    #pragma omp parallel for
    for (int ia = 0; ia < natom; ia++) {
        int p0 = gslice_by_atom[ia * 2];
        int p1 = gslice_by_atom[ia * 2 + 1];
        
        // Validate grid slice bounds
        if (p0 < 0 || p1 > ngrids || p0 >= p1) {
            continue; // Skip invalid slice
        }
        
        int ncoords = p1 - p0;
        
        // Allocate temporary arrays for this thread
        double *ri_rJ = malloc(ncoords * natom * 3 * sizeof(double));
        double *riJ = malloc(ncoords * natom * sizeof(double));
        double *diJ = malloc(ncoords * natom * sizeof(double));
        double *fiJ = malloc(ncoords * natom * sizeof(double));
        double *dfiJ = malloc(ncoords * natom * 3 * sizeof(double));
        
        if (!ri_rJ || !riJ || !diJ || !fiJ || !dfiJ) {
            // Memory allocation failed
            free(ri_rJ); free(riJ); free(diJ); free(fiJ); free(dfiJ);
            continue;
        }
        
        // Calculate ri_rJ = grid_coords[p0:p1] - atom_coords for all atoms
        for (int i = 0; i < ncoords; i++) {
            for (int ja = 0; ja < natom; ja++) {
                int grid_idx = p0 + i;
                int ri_rJ_idx = i * natom * 3 + ja * 3;
                
                ri_rJ[ri_rJ_idx + 0] = grid_coords[grid_idx * 3 + 0] - atom_coords[ja * 3 + 0];
                ri_rJ[ri_rJ_idx + 1] = grid_coords[grid_idx * 3 + 1] - atom_coords[ja * 3 + 1];
                ri_rJ[ri_rJ_idx + 2] = grid_coords[grid_idx * 3 + 2] - atom_coords[ja * 3 + 2];
                
                // Calculate distance riJ
                riJ[i * natom + ja] = sqrt(ri_rJ[ri_rJ_idx + 0] * ri_rJ[ri_rJ_idx + 0] +
                                         ri_rJ[ri_rJ_idx + 1] * ri_rJ[ri_rJ_idx + 1] +
                                         ri_rJ[ri_rJ_idx + 2] * ri_rJ[ri_rJ_idx + 2]);
            }
        }
        
        // Calculate diJ = (riJ - R_in_J) / R_sw_J
        for (int i = 0; i < ncoords; i++) {
            for (int ja = 0; ja < natom; ja++) {
                int idx = i * natom + ja;
                diJ[idx] = (riJ[idx] - R_in_J[ja]) / R_sw_J[ja];
                
                // Set diJ[:, ia] = 1.0
                if (ja == ia) {
                    diJ[idx] = 1.0;
                }
                
                // Set small values to zero
                if (diJ[idx] < 1e-8) {
                    diJ[idx] = 0.0;
                    // Set ri_rJ to zero for small diJ values
                    int ri_rJ_idx = i * natom * 3 + ja * 3;
                    ri_rJ[ri_rJ_idx + 0] = 0.0;
                    ri_rJ[ri_rJ_idx + 1] = 0.0;
                    ri_rJ[ri_rJ_idx + 2] = 0.0;
                }
            }
            
            // Set ri_rJ[:, ia, :] = 0.0
            int ri_rJ_idx = i * natom * 3 + ia * 3;
            ri_rJ[ri_rJ_idx + 0] = 0.0;
            ri_rJ[ri_rJ_idx + 1] = 0.0;
            ri_rJ[ri_rJ_idx + 2] = 0.0;
        }
        
        // Calculate fiJ and dfiJ
        for (int i = 0; i < ncoords; i++) {
            for (int ja = 0; ja < natom; ja++) {
                int idx = i * natom + ja;
                fiJ[idx] = switch_h(diJ[idx]);
                double grad_h = grad_switch_h(diJ[idx]);
                
                // Calculate dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J) * ri_rJ
                // Handle division by zero carefully
                double factor = 0.0;
                if (fabs(grad_h) > 1e-15 && fabs(fiJ[idx]) > 1e-15 && 
                    fabs(riJ[idx]) > 1e-15 && fabs(R_sw_J[ja]) > 1e-15) {
                    factor = grad_h / (fiJ[idx] * riJ[idx] * R_sw_J[ja]);
                }
                
                int ri_rJ_idx = i * natom * 3 + ja * 3;
                int dfiJ_idx = i * natom * 3 + ja * 3;
                dfiJ[dfiJ_idx + 0] = factor * ri_rJ[ri_rJ_idx + 0];
                dfiJ[dfiJ_idx + 1] = factor * ri_rJ[ri_rJ_idx + 1];
                dfiJ[dfiJ_idx + 2] = factor * ri_rJ[ri_rJ_idx + 2];
            }
        }
        
        // Calculate grid response: dFi_grid = sum(dfiJ, axis=1)
        for (int i = 0; i < ncoords; i++) {
            int grid_idx = p0 + i;
            double Fi = switch_fun[grid_idx];
            double Ai = area[grid_idx];
            
            double dFi_grid[3] = {0.0, 0.0, 0.0};
            for (int ja = 0; ja < natom; ja++) {
                int dfiJ_idx = i * natom * 3 + ja * 3;
                dFi_grid[0] += dfiJ[dfiJ_idx + 0];
                dFi_grid[1] += dfiJ[dfiJ_idx + 1];
                dFi_grid[2] += dfiJ[dfiJ_idx + 2];
            }
            
            // Update dF[p0:p1, ia, :] += Fi * dFi_grid
            // Update dA[p0:p1, ia, :] += Ai * dFi_grid
            int df_idx = grid_idx * natom * 3 + ia * 3;
            dF[df_idx + 0] += Fi * dFi_grid[0];
            dF[df_idx + 1] += Fi * dFi_grid[1];
            dF[df_idx + 2] += Fi * dFi_grid[2];
            
            dA[df_idx + 0] += Ai * dFi_grid[0];
            dA[df_idx + 1] += Ai * dFi_grid[1];
            dA[df_idx + 2] += Ai * dFi_grid[2];
        }
        
        // Calculate atom response
        for (int i = 0; i < ncoords; i++) {
            int grid_idx = p0 + i;
            double Fi = switch_fun[grid_idx];
            double Ai = area[grid_idx];
            
            for (int ja = 0; ja < natom; ja++) {
                int dfiJ_idx = i * natom * 3 + ja * 3;
                int df_idx = grid_idx * natom * 3 + ja * 3;
                
                // Update dF[p0:p1, :, :] -= Fi * dfiJ
                // Update dA[p0:p1, :, :] -= Ai * dfiJ
                dF[df_idx + 0] -= Fi * dfiJ[dfiJ_idx + 0];
                dF[df_idx + 1] -= Fi * dfiJ[dfiJ_idx + 1];
                dF[df_idx + 2] -= Fi * dfiJ[dfiJ_idx + 2];
                
                dA[df_idx + 0] -= Ai * dfiJ[dfiJ_idx + 0];
                dA[df_idx + 1] -= Ai * dfiJ[dfiJ_idx + 1];
                dA[df_idx + 2] -= Ai * dfiJ[dfiJ_idx + 2];
            }
        }
        
        // Clean up temporary arrays
        free(ri_rJ);
        free(riJ);
        free(diJ);
        free(fiJ);
        free(dfiJ);
    }
    
    return 0; // Success
}

/*
 * Wrapper function that can be called from Python via ctypes
 * All arrays are assumed to be contiguous and in C order
 */
int PCM_get_dF_dA_wrapper(double *atom_coords, double *grid_coords,
                         double *switch_fun, double *area,
                         double *R_in_J, double *R_sw_J, 
                         int *gslice_by_atom,
                         int natom, int ngrids,
                         double *dF, double *dA) {
    return PCM_get_dF_dA(atom_coords, grid_coords, switch_fun, area,
                        R_in_J, R_sw_J, gslice_by_atom,
                        natom, ngrids, dF, dA);
}