import numpy as np
import os
import sys
import h5py

from itertools import product as itp
import gem
from scipy.sparse.linalg import eigsh

def get_group_keys(data_path: str) -> str:
    with h5py.File(data_path, 'r') as file:
        return list(file.keys())

def get_group(data_path: str, group_name: str):
    with h5py.File(data_path, 'r') as file:
        return np.array(file.get(group_name))

def get_subgroup(data_path: str, group_name: str, subgroup_name: str):
    """
    Retrieve a specified subgroup within a given group from an HDF5 file.

    Parameters:
        data_path (str): Path to the HDF5 file.
        group_name (str): Name of the group containing the desired subgroup.
        subgroup_name (str): Name of the subgroup to retrieve.

    Returns:
        np.ndarray: The data in the specified subgroup as a NumPy array.
    """
    with h5py.File(data_path, 'r',  libver='latest', swmr=True) as file:
        group = file.get(group_name)
        if group is None:
            raise KeyError(f"Group '{group_name}' not found in the file.")
        subgroup = group.get(subgroup_name)
        if subgroup is None:
            raise KeyError(f"Subgroup '{subgroup_name}' not found in group '{group_name}'.")
        return np.array(subgroup)

def get_group_column(data_path: str, group_name: str, column_idx: int) -> np.ndarray:
    """
    Reads a specific column of a group dataset in an HDF5 file.

    Args:
        data_path (str): Path to the HDF5 file.
        group_name (str): Name of the group/dataset.
        column_idx (int): Index of the column to extract.

    Returns:
        np.ndarray: The specified column as a numpy array.
    """
    with h5py.File(data_path, 'r') as file:
        dataset = file[group_name]
        if column_idx >= dataset.shape[1]:  # Check if the column index is valid
            raise IndexError(f"Column index {column_idx} is out of bounds for dataset with shape {dataset.shape}")
        column = dataset[:, column_idx]  # Read only the specified column
    return column

def get_Umatrix_columns(OVERLAP_PATH: str, COLUMNS_IDXS: list[int]) -> list[np.ndarray]:
    """
    Returns specific columns of the U_hat matrix given their indices.

    Args:
        OVERLAP_PATH (str): Path to the HDF5 file.
        COLUMNS_IDXS (list[int]): List of column indices to extract.

    Returns:
        list[np.ndarray]: A list containing the specified columns as numpy arrays.
    """
    print("Requested indices:", COLUMNS_IDXS)
    columns = []
    for idx in COLUMNS_IDXS:
        try:
            column = get_group_column(OVERLAP_PATH, "U_hat", idx)
            columns.append(column)
        except IndexError as e:
            print(f"Error: {e}")
    return columns

def construct_HtildeX(Htilde_onebody_path: str,
                      Htilde_manybody_path: str,
                      Xparams: np.ndarray,
                      bath_orbs: int
                      )-> np.ndarray:
    '''
    Constructs the projected EH in the variational space spanned by {|u>}
        Htilde_onebody_path: path to projected onebody EH terms
        Htilde_manybody_path: path to projected manybody/interaction EH terms
    Note: sz_pen = 0 and spin_pen = 10
    '''
    sz_pen = 0
    spin_pen = 10
    Htilde_one_keys = get_group_keys(Htilde_onebody_path)
    Htilde_one = get_group(Htilde_onebody_path, Htilde_one_keys[0])
    Htilde_one += sum(Xparams[r] * get_group(Htilde_onebody_path, Htilde_one_keys[r+1]) for r in range(bath_orbs))
    Htilde_int_terms = [get_group(Htilde_manybody_path, key) for key in get_group_keys(Htilde_manybody_path)]
    Htilde_X = Htilde_one + Htilde_int_terms[0] + spin_pen*Htilde_int_terms[1] + sz_pen*Htilde_int_terms[2].dot(Htilde_int_terms[2])

    return Htilde_X


def solve_Htilde(Htilde_X: np.ndarray,
                  K: int):
    if K is None:
        print("- In solve_Htilde -- K: None and dim(H) = %d" % (len(Htilde_X)))
    else:
        print("- In solve_Htilde -- K: %d and dim(H) = %d" % (K, len(Htilde_X)))

    assert(K is None or K <= len(Htilde_X))
    Htilde_K = Htilde_X[:K, :K].real
    print("shape of Htilde_K: ", Htilde_K.shape)

    vals, vecs = eigsh(Htilde_K, k=2, which='SA', tol=1e-12)
    Etilde = vals[0]
    psi_K = vecs[:, 0]
    return Etilde, psi_K

def construct_DMtilde(approx_GS: np.ndarray,
                      total_orbs: int,
                      DM_op,
                      K: int):
    if K is None:
        print("- In construct_DMtilde -- K: None")
    else:
        print("- In construct_DMtilde -- K: %d" % K)

    psi_K  = approx_GS # redefine for convenience
    psi_dag = psi_K.conj().T

    if DM_op is None:
        raise ValueError("DM_op is None. Use load_DM_op.")

    DM_K = np.zeros((total_orbs, total_orbs), dtype = np.complex128 )
    for a in range(total_orbs):
        for b in range(a,total_orbs):
            DM_K[a,b] = psi_dag.dot(DM_op[a, b].dot(psi_K))
            if a != b:
                DM_K[b,a] = DM_K[a,b].conjugate()
    return DM_K



class SVDSolver2(object):
    ''' SVD solver class'''
    def __init__(self, ntot, nimp, nbath, suff="", K=None, solver_params=None):
        """Constructor method
        """
        self.type = "SVDSolver"
        self.solver_params = solver_params if solver_params is not None else {}
        self.ntot = ntot
        self.nimp = nimp
        self.nbath = nbath
        # self.set_kwargs(params)
        self.suff = suff
        self.gs_ene = 0
        self.denMat = 0

        self.K = K
        self.DM_op = None

        self.num_h = 0
        self.Htilde_r = []
        self.M_list = []
        self.X = 0
        self.v_all = 0
        self.shift = 0
        self.U = 1

#MANDATORY FUNCTIONS
    def build_Hemb(self, D, H1E, Lambda, V2E):

        ntot, nimp, nbath = self.ntot, self.nimp, self.nbath

        full_M = np.zeros((ntot, ntot), dtype=np.complex128)
        full_M[:nimp, :nimp] = H1E
        full_M[:nimp, nimp:] = np.conjugate(D.T)
        full_M[nimp:, :nimp] = D
        full_M[nimp:, nimp:] = -Lambda

        print(V2E)
        self.U = V2E[0, 0, 0, 0]

        print("full M")
        print(np.around(full_M, 10))

        full_M = full_M/self.U

        self.shift = (full_M[0, 0] + full_M[0, 0] + 1)/2

        full_M = full_M.real - self.shift * np.eye(ntot)

        M = full_M
        print("full M/U - shift")
        print(np.around(M, 10))

        v_s = {}
        R_s = {}

        B_s = {}
        W_s = {}

        for s, spin in enumerate(["up", "dn"]):

            print("Diagonalization of the bath, spin %s" % spin)
            M = full_M[s::2, s::2]
            B = M[nimp//2:, nimp//2:]
            w, v = np.linalg.eigh(B)
            # print("v")
            # print(v)
            W = M[:nimp//2, nimp//2:]
            W_s[spin] = W
            W_rot = W @ v

            R = np.eye((nbath//2))
            for r in range(nbath//2):
                if W_rot[0][r] > 0:
                    R[r, r] = -1
            R_s[spin] = R
            v = v @ R

            # print("W")
            # print(W)
            # print(W_rot)

            B_s[spin] = B
            B_rot = v.T.conjugate() @ B @ v
            # print("B")
            # print(B)
            # print(B_rot)

            ind = np.argsort(np.diag(B_rot))[::-1]
            B_rot[:] = B_rot[:, ind]
            print(B_rot)
            v[:, :] = v[:, ind]

            # ind = np.argsort(np.diag(B_rot))
            # v[:, :] = v[:, ind]


            v_s[spin] = np.block([[np.eye(nimp//2), np.zeros((nimp//2, nbath*nimp//4))],
                                  [np.zeros((nbath*nimp//4, nimp//2)), v]])
        # print("diff B")
        # print(B_s["up"] - B_s["dn"])
        # print("diff W")
        # print(W_s["up"] - W_s["dn"])
        # print("diff v")
        # print(v_s["up"] - v_s["dn"])

        self.v_all = np.zeros((ntot, ntot), dtype=np.complex128)
        self.v_all[::2, ::2] = v_s["up"]    # @ R_s["up"]
        self.v_all[1::2, 1::2] = v_s["dn"]  # @ R_s["dn"]

        M_rot = (self.v_all.T.conjugate() @ full_M @ self.v_all)

        print("M_rot")
        print(np.around(M_rot, 8).real)
        M_rot = M_rot.real

        self.X = list(np.diag(-M_rot[2::2, 2::2])) + list(M_rot[2::2, 0])
        print("X: ", self.X)
        print()


    def solve_Hemb(self, num_eig=None, verbose=None, T=0.0):

        X = self.X
        nbath = self.nbath

        Htilde_X = construct_HtildeX(self.Htilde_one_path, self.Htilde_int_path, X, nbath)
        self.gs_ene, groundstate = solve_Htilde(Htilde_X, self.K)
        self.gs_ene = self.gs_ene * self.U
        self.gs_wf = groundstate
        print(" Ground state energy: %.12f" % self.gs_ene)

        svd_dm = construct_DMtilde(groundstate, self.ntot, self.DM_op, self.K)
        self.denMat = self.v_all @ svd_dm @ self.v_all.T.conjugate()
        # print(np.around(svd_dm.real, 5))
        # print(np.around(self.v_all.real, 5))
        # print(np.around(self.v_all.T.conjugate().real, 5))
        # print(np.around(self.denMat.real, 5))

    def calc_density_matrix(self):
        return self.denMat.real

    def compute_E2loc(self):
        print('warning: E2loc not implemented')
        return 0

#AUXILIARY FUNCTIONS
    def load_stuff(self, path):
        nbath = self.nbath
        ntot = self.ntot
        K = self.K

        # Define paths relative to the root directory
        self.Htilde_one_path = os.path.join(path, f"data/Htilde_one_B{nbath//2}.h5")
        self.Htilde_int_path = os.path.join(path, f"data/Htilde_int_B{nbath//2}.h5")
        self.DenMat_tilde_path = os.path.join(path, f"data/DM_project_B{nbath//2}.h5")

        self.DM_op = np.zeros((ntot, ntot, K, K), dtype = np.complex128)
        for a in range(ntot):
            for b in range(a, ntot):
                self.DM_op[a, b] = get_group(self.DenMat_tilde_path, f"op_{a}_{b}")[:K,:K]

    def calc_double_occ(self,idx):
        H_int = get_group(self.Htilde_int_path, "Htilde_int_1")[:self.K, :self.K]
        psi_K = self.gs_wf
        psi_dag = psi_K.conj().T
        self.docc = psi_dag.dot(H_int.dot(psi_K))
        # print('warning: double occupancy not implement!')
        return self.docc
