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

import unittest
import numpy
from numpy import testing
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.scf import khf
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import describe_nested, nested_to_vector, vector_to_nested

class KnownValues(unittest.TestCase):
    def test_kconserve(self):
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.build()
        kpts = cell.make_kpts([3,4,5])
        kconserve = tools.get_kconserv(cell, kpts)
        self.assertAlmostEqual(lib.finger(kconserve), 84.88659638289468, 9)

    def test_kconserve3(self):
        cell = pbcgto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.build()
        kpts = cell.make_kpts([2,2,2])
        nkpts = kpts.shape[0]
        kijkab = [range(nkpts),range(nkpts),1,range(nkpts),range(nkpts)]
        kconserve = tools.get_kconserv3(cell, kpts, kijkab)
        self.assertAlmostEqual(lib.finger(kconserve), -3.1172758206126852, 0)


class TestVecNested(unittest.TestCase):
    def test_1(self):
        struct = (
            (numpy.random.rand(3, 1, 4), numpy.zeros(5)),
            (numpy.random.rand(9),),
        )
        vec, desc = nested_to_vector(struct)
        self.assertEqual(desc, [
            [
                dict(type="array", shape=(3, 1, 4)),
                dict(type="array", shape=(5,)),
            ],
            [
                dict(type="array", shape=(9,)),
            ]
        ])
        self.assertEqual(len(vec), 3 * 4 + 5 + 9)
        struct_recovered = vector_to_nested(vec, desc)
        testing.assert_equal(struct, struct_recovered)

    def test_composite(self):
        struct = (
            (numpy.random.rand(3, 1, 4), numpy.zeros(5)),
            (numpy.random.rand(9),),
        )
        vec, desc = nested_to_vector(struct)

        # Test orig
        desc = ((desc[0][0], desc[0][1]), desc[1])
        struct_recovered = vector_to_nested(vec, desc)
        testing.assert_equal(struct, struct_recovered)

        # Test modif
        desc = ((dict(
            type="composite",
            dtype=float,
            shape=(3, 1, 4),
            data=(
                dict(type="array", shape=(1, 4)),
            ) * 3,
        ), desc[0][1]), desc[1])
        struct_recovered = vector_to_nested(vec, desc)
        testing.assert_equal(struct, struct_recovered)
