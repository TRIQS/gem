################################################################################
#
# GEM : Ghost Embedding Method
#
# Copyright (C) 2026, The Simons Foundation
#   author: S. Giuli
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

r"""
Module containing GEM : Ghost Embedding Method

"""

import gem.logo
from gem.mpi import resolve_comm

if resolve_comm(None).Get_rank() == 0:
    print(gem.logo.logo())
