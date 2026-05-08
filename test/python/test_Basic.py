#!/usr/bin/env python

import unittest

from triqs_ghostGA.gdmft import Gdmft
from triqs_ghostGA.fragment import Fragment
from triqs_ghostGA.lattice import Lattice
from triqs_ghostGA.version import show_version, show_git_hash


class test_basic_features(unittest.TestCase):

    def test_loading(self):
        assert Gdmft is not None
        assert Fragment is not None
        assert Lattice is not None

    def test_version_prints(self):
        show_version()
        show_git_hash()


if __name__ == '__main__':
    unittest.main()
