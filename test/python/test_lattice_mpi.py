import unittest
import numpy as np

from gem.lattice import Lattice
from gem.mpi import split_range


class TestSplitRange(unittest.TestCase):
    '''The k-point split must be a partition, whatever nk and size are.'''

    def test_partition(self):
        for n in [0, 1, 2, 5, 101, 5001]:
            for size in [1, 2, 3, 4, 7, 8, 16]:
                blocks = [split_range(n, r, size) for r in range(size)]
                # contiguous, non-overlapping, covering range(n)
                self.assertEqual(blocks[0][0], 0)
                self.assertEqual(blocks[-1][1], n)
                for (s0, e0), (s1, _) in zip(blocks, blocks[1:]):
                    self.assertLessEqual(s0, e0)
                    self.assertEqual(e0, s1)
                # balanced to within one element
                sizes = [e - s for s, e in blocks]
                self.assertEqual(sum(sizes), n)
                self.assertLessEqual(max(sizes) - min(sizes), 1)


class TestLatticeSerialPath(unittest.TestCase):
    '''use_mpi=False must leave every rank with the whole k-range.'''

    def setUp(self):
        nk = 51
        e = np.linspace(-1, 1, nk)
        self.wks = np.sqrt(1 - e**2)
        self.wks /= np.sum(self.wks)
        self.eks = np.array([np.array([[1.0*x]], dtype=np.complex128) for x in e])

    def test_full_range_and_weights(self):
        lat = Lattice(self.eks, wk_list=self.wks, use_mpi=False)
        self.assertEqual((lat.k0, lat.k1), (0, len(self.wks)))
        self.assertEqual(lat.mpi_size, 1)
        self.assertEqual(len(list(lat._my_ks())), len(self.wks))

    def test_eks_not_sliced(self):
        # consumers outside the class iterate lat.eks/lat.wks directly
        lat = Lattice(self.eks, wk_list=self.wks)
        self.assertEqual(lat.eks.shape, self.eks.shape)
        self.assertEqual(lat.wks.shape, self.wks.shape)
        np.testing.assert_array_equal(lat.wks, self.wks)

    def test_unnormalised_weights_rejected(self):
        # the k sums carry no 1/sum(wk), so unnormalised weights would give a
        # correct density and a free energy wrong by that factor
        with self.assertRaises(ValueError):
            Lattice(self.eks, wk_list=2.0*self.wks)
        with self.assertRaises(ValueError):
            Lattice(self.eks, wk_list=np.ones(len(self.wks)))
        # the default weights, and anything within round-off of 1, are fine
        Lattice(self.eks)
        Lattice(self.eks, wk_list=self.wks*(1 + 1e-12))


if __name__ == '__main__':
    unittest.main()
