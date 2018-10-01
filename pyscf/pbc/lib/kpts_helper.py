#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import itertools
from collections import OrderedDict
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf import __config__

KPT_DIFF_TOL = getattr(__config__, 'pbc_lib_kpts_helper_kpt_diff_tol', 1e-6)


def is_zero(kpt):
    return abs(np.asarray(kpt)).sum() < KPT_DIFF_TOL
gamma_point = is_zero

def member(kpt, kpts):
    kpts = np.reshape(kpts, (len(kpts),kpt.size))
    dk = np.einsum('ki->k', abs(kpts-kpt.ravel()))
    return np.where(dk < KPT_DIFF_TOL)[0]

def unique(kpts):
    kpts = np.asarray(kpts)
    nkpts = len(kpts)
    uniq_kpts = []
    uniq_index = []
    uniq_inverse = np.zeros(nkpts, dtype=int)
    seen = np.zeros(nkpts, dtype=bool)
    n = 0
    for i, kpt in enumerate(kpts):
        if not seen[i]:
            uniq_kpts.append(kpt)
            uniq_index.append(i)
            idx = abs(kpt-kpts).sum(axis=1) < KPT_DIFF_TOL
            uniq_inverse[idx] = n
            seen[idx] = True
            n += 1
    return np.asarray(uniq_kpts), np.asarray(uniq_index), uniq_inverse

def loop_kkk(nkpts):
    range_nkpts = range(nkpts)
    return itertools.product(range_nkpts, range_nkpts, range_nkpts)

def get_kconserv(cell, kpts):
    r'''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satifies momentum conservation,

        (k(k) - k(l) + k(m) - k(n)) \dot a = 2n\pi

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2*np.pi)

    kconserv = np.zeros((nkpts,nkpts,nkpts), dtype=int)
    kvMLK = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    for N, kvN in enumerate(kpts):
        kvMLKN = np.einsum('klmx,wx->mlkw', kvMLK - kvN, a)
        # check whether (1/(2pi) k_{KLMN} dot a) is an integer
        kvMLKN_int = np.rint(kvMLKN)
        mask = np.einsum('klmw->mlk', abs(kvMLKN - kvMLKN_int)) < 1e-9
        kconserv[mask] = N
    return kconserv


    if kconserv is None:
        kconserv = get_kconserv(cell, kpts)

    arr_offset = []
    arr_size = []
    offset = 0
    for kk, kl, km in loop_kkk(nkpts):
        kn = kconserv[kk, kl, km]

        # Get array size for these k-points and add offset
        size = np.prod([norb_per_kpt[x] for x in [kk, kl, km, kn]])

        arr_size.append(size)
        arr_offset.append(offset)

        offset += size
    return arr_offset, arr_size, (arr_size[-1] + arr_offset[-1])


def check_kpt_antiperm_symmetry(array, idx1, idx2, tolerance=1e-8):
    '''Checks antipermutational symmetry for k-point array.

    Checks whether an array with k-point symmetry has antipermutational symmetry
    with respect to switching the particle indices `idx1`, `idx2`. The particle
    indices switches both the orbital index and k-point index associated with
    the two indices.

    Note:
        One common reason for not obeying antipermutational symmetry in a calculation
        involving FFTs is that the grid to perform the FFT may be too coarse.  This
        symmetry is present in operators in spin-orbital form and 'spin-free'
        operators.

    array (:obj:`ndarray`): array to test permutational symmetry, where for
        an n-particle array, the first (2n-1) array elements are kpoint indices
        while the final 2n array elements are orbital indices.
    idx1 (int): first index
    idx2 (int): second index

    Examples:
        For a 3-particle array, such as the T3 amplitude
            t3[ki, kj, kk, ka, kb, i, j, a, b, c],
        setting `idx1 = 0` and `idx2 = 1` would switch the orbital indices i, j as well
        as the kpoint indices ki, kj.

        >>> nkpts, nocc, nvir = 3, 4, 5
        >>> t2 = numpy.random.random_sample((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir))
        >>> t2 = t2 + t2.transpose(1,0,2,4,3,5,6)
        >>> check_kpt_antiperm_symmetry(t2, 0, 1)
        True
    '''
    # Checking to make sure bounds of idx1 and idx2 are O.K.
    assert(idx1 >= 0 and idx2 >= 0 and 'indices to swap must be non-negative!')

    array_shape_len = len(array.shape)
    nparticles = (array_shape_len + 1) / 4
    assert(idx1 < (2 * nparticles - 1) and idx2 < (2 * nparticles - 1) and
           'This function does not support the swapping of the last k-point index '
           '(This k-point is implicitly not indexed due to conservation of momentum '
           'between k-points.).')

    if (nparticles > 3):
        raise NotImplementedError('Currently set up for only up to 3 particle '
                                  'arrays. Input array has %d particles.')

    kpt_idx1 = idx1
    kpt_idx2 = idx2

    # Start of the orbital index, located after k-point indices
    orb_idx1 = (2 * nparticles - 1) + idx1
    orb_idx2 = (2 * nparticles - 1) + idx2

    # Sign of permutation
    sign = (-1)**(abs(idx1 - idx2) + 1)
    out_array_indices = np.arange(array_shape_len)

    out_array_indices[kpt_idx1], out_array_indices[kpt_idx2] = \
            out_array_indices[kpt_idx2], out_array_indices[kpt_idx1]
    out_array_indices[orb_idx1], out_array_indices[orb_idx2] = \
            out_array_indices[orb_idx2], out_array_indices[orb_idx1]
    antisymmetric = (np.linalg.norm(array + array.transpose(out_array_indices)) <
                     tolerance)
    return antisymmetric


def get_kconserv3(cell, kpts, kijkab):
    '''Get the momentum conservation array for a set of k-points.

    This function is similar to get_kconserv, but instead finds the 'kc'
    that satisfies momentum conservation for 5 k-points,

        (ki + kj + kk - ka - kb - kc) dot a = 2n\pi

    where these kpoints are stored in kijkab[ki, kj, kk, ka, kb].
    '''
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2*np.pi)

    kpts_i, kpts_j, kpts_k, kpts_a, kpts_b = \
            [kpts[x].reshape(-1,3) for x in kijkab]
    shape = [np.size(x) for x in kijkab]
    kconserv = np.zeros(shape, dtype=int)

    kv_kab = kpts_k[:,None,None,:] - kpts_a[:,None,:] - kpts_b
    for i, kpti in enumerate(kpts_i):
        for j, kptj in enumerate(kpts_j):
            kv_ijkab = kv_kab + kpti + kptj
            for c, kptc in enumerate(kpts):
                s = np.einsum('kabx,wx->kabw', kv_ijkab - kptc, a)
                s_int = np.rint(s)
                mask = np.einsum('kabw->kab', abs(s - s_int)) < 1e-9
                kconserv[i,j,mask] = c

    new_shape = [shape[i] for i, x in enumerate(kijkab)
                 if not isinstance(x, (int,np.int))]
    kconserv = kconserv.reshape(new_shape)
    return kconserv


def describe_nested(data):
    """
    Retrieves the description of a nested array structure.
    Args:
        data (iterable): a nested structure to describe;

    Returns:
        - A nested structure where numpy arrays are replaced by their shapes;
        - The overall number of scalar elements;
        - The common data type;
    """
    if isinstance(data, np.ndarray):
        return dict(
            type="array",
            shape=data.shape,
        ), data.size, data.dtype
    elif isinstance(data, (list, tuple)):
        total_size = 0
        struct = []
        dtype = None
        for i in data:
            i_struct, i_size, i_dtype = describe_nested(i)
            struct.append(i_struct)
            total_size += i_size
            if dtype is not None and i_dtype is not None and i_dtype != dtype:
                raise ValueError("Several different numpy dtypes encountered: %s and %s" %
                                 (str(dtype), str(i_dtype)))
            dtype = i_dtype
        return struct, total_size, dtype
    else:
        raise ValueError("Unknown object to describe: %s" % str(data))


def nested_to_vector(data, destination=None, offset=0):
    """
    Puts any nested iterable into a vector.
    Args:
        data (Iterable): a nested structure of numpy arrays;
        destination (array): array to store the data to;
        offset (int): array offset;

    Returns:
        If destination is not specified, returns a vectorized data and the original nested structure to restore the data
        into its original form. Otherwise returns a new offset.
    """
    if destination is None:
        struct, total_size, dtype = describe_nested(data)
        destination = np.empty(total_size, dtype=dtype)
        rtn = True
    else:
        rtn = False

    if isinstance(data, np.ndarray):
        destination[offset:offset + data.size] = data.ravel()
        offset += data.size
    elif isinstance(data, (list, tuple)):
        for i in data:
            offset = nested_to_vector(i, destination, offset)
    else:
        raise ValueError("Unknown object to vectorize: %s" % str(data))

    if rtn:
        return destination, struct
    else:
        return offset


def vector_to_nested(vector, struct, copy=True, ensure_size_matches=True, destination=None, destination_indexes=None):
    """
    Retrieves the original nested structure from the vector.
    Args:
        vector (ndarray): a vector to decompose;
        struct (Iterable): a nested structure with arrays' shapes;
        copy (bool): whether to copy arrays;
        ensure_size_matches (bool): if True, ensures all elements from the vector are used;
        destination (ndarray): an array to write to;
        destination_indexes (Iterable): first indexes to the array;

    Returns:
        A nested structure with numpy arrays and, if `ensure_size_matches=False`, the number of vector elements used.
    """
    if len(vector.shape) != 1:
        raise ValueError("Only vectors accepted, got: %s" % repr(vector.shape))

    if destination_indexes is None:
        destination_indexes = tuple()

    if isinstance(struct, dict):

        if "type" not in struct:
            raise ValueError("Missing 'type' key in structure: {}".format(struct))

        # Case: array
        if struct["type"] == "array":

            shape = struct["shape"]
            expected_size = np.prod(shape)
            if ensure_size_matches:
                if vector.size != expected_size:
                    raise ValueError("Structure size mismatch: expected %s = %d, found %d" %
                                     (repr(shape), expected_size, vector.size,))
            if len(vector) < expected_size:
                raise ValueError("Additional %d = (%d = %s) - %d vector elements are required" %
                                 (expected_size - len(vector), expected_size,
                                  repr(shape), len(vector),))

            if destination is None:
                a = vector[:expected_size].reshape(shape)
                if copy:
                    a = a.copy()
                if ensure_size_matches:
                    return a
                else:
                    return a, expected_size

            else:
                if shape != destination.shape[len(destination_indexes):]:
                    raise ValueError("Composite array shape mismatch: expected %s, found %s" %
                                     (shape, destination.shape[len(destination_indexes):]))
                destination[np.ix_(*destination_indexes)] = vector[:expected_size].reshape(shape)
                if ensure_size_matches:
                    return destination
                else:
                    return destination, expected_size

        # Case: composite
        elif struct["type"] == "composite":

            shape = struct["shape"]
            dtype = struct["dtype"]
            underlying_struct = struct["data"]
            if destination is not None:
                if shape != destination.shape[len(destination_indexes):]:
                    raise ValueError("Composite array shape mismatch: expected %s, found %s" %
                                     (shape, destination.shape[len(destination_indexes):]))
                if dtype != destination.dtype:
                    raise ValueError("Dtype mismatch: expected %s, found %s" % (destination.dtype, dtype))
            else:
                destination = np.zeros(struct["shape"], dtype=struct["dtype"])
                destination_indexes = tuple()

            return vector_to_nested(vector, underlying_struct,
                                    ensure_size_matches=ensure_size_matches,
                                    destination=destination,
                                    destination_indexes=destination_indexes
                                    )

        else:
            raise ValueError("Unknown structure type: {}".format(struct["type"]))

    # Case: nested
    elif isinstance(struct, (list, tuple)):

        if destination is None:
            offset = 0
            result = []
            for i in struct:
                nested, size = vector_to_nested(vector[offset:], i, copy=copy, ensure_size_matches=False)
                offset += size
                result.append(nested)

            if ensure_size_matches:
                if vector.size != offset:
                    raise ValueError("%d additional elements found" % (vector.size - offset))
                return result
            else:
                return result, offset

        else:
            expected_len = destination.shape[len(destination_indexes)]
            if len(struct) != expected_len:
                raise ValueError("Nested length mismatch: expected %d, found %d" % (expected_len, len(struct)))

            offset = 0
            for i, element in enumerate(struct):
                offset += vector_to_nested(
                    vector[offset:],
                    element,
                    ensure_size_matches=False,
                    destination=destination,
                    destination_indexes=destination_indexes + ((i,),),
                )

            if ensure_size_matches:
                if vector.size != offset:
                    raise ValueError("%d additional elements found" % (vector.size - offset))
                return destination
            else:
                return destination, offset

    else:
        raise ValueError("Unknown object to compose: %s" % (str(struct)))


class KptsHelper(lib.StreamObject):
    def __init__(self, cell, kpts):
        '''Helper class for handling k-points in correlated calculations.

        Attributes:
            kconserv : (nkpts,nkpts,nkpts) ndarray
                The index of the fourth momentum-conserving k-point, given
                indices of three k-points
            symm_map : OrderedDict of list of (3,) tuples
                Keys are (3,) tuples of symmetry-unique k-point indices and
                values are lists of (3,) tuples, enumerating all
                symmetry-related k-point indices for ERI generation
        '''
        self.kconserv = get_kconserv(cell, kpts)
        nkpts = len(kpts)
        temp = range(0,nkpts)
        kptlist = lib.cartesian_prod((temp,temp,temp))
        completed = np.zeros((nkpts,nkpts,nkpts), dtype=bool)

        self._operation = np.zeros((nkpts,nkpts,nkpts), dtype=int)
        self.symm_map = OrderedDict()

        for kpt in kptlist:
            kpt = tuple(kpt)
            kp,kq,kr = kpt
            if not completed[kp,kq,kr]:
                self.symm_map[kpt] = list()
                ks = self.kconserv[kp,kq,kr]

                completed[kp,kq,kr] = True
                self._operation[kp,kq,kr] = 0
                self.symm_map[kpt].append((kp,kq,kr))

                completed[kr,ks,kp] = True
                self._operation[kr,ks,kp] = 1 #.transpose(2,3,0,1)
                self.symm_map[kpt].append((kr,ks,kp))

                completed[kq,kp,ks] = True
                self._operation[kq,kp,ks] = 2 #np.conj(.transpose(1,0,3,2))
                self.symm_map[kpt].append((kq,kp,ks))

                completed[ks,kr,kq] = True
                self._operation[ks,kr,kq] = 3 #np.conj(.transpose(3,2,1,0))
                self.symm_map[kpt].append((ks,kr,kq))


    def transform_symm(self, eri_kpt, kp, kq, kr):
        '''Return the symmetry-related ERI at any set of k-points.

        Args:
            eri_kpt : (nmo,nmo,nmo,nmo) ndarray
                An in-cell ERI calculated with a set of symmetry-unique k-points.
            kp, kq, kr : int
                The indices of the k-points at which the ERI is desired.
        '''
        operation = self._operation[kp,kq,kr]
        if operation == 0:
            return eri_kpt
        if operation == 1:
            return eri_kpt.transpose(2,3,0,1)
        if operation == 2:
            return np.conj(eri_kpt.transpose(1,0,3,2))
        if operation == 3:
            return np.conj(eri_kpt.transpose(3,2,1,0))

