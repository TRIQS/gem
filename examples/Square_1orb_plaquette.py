import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
import matplotlib.pyplot as plt

# =============================================================
# Square lattice Hubbard model — single 2x2 plaquette fragment
#
# The unit cell is a 2x2 plaquette with site positions (in units
# of the original lattice constant):
#   s0=(0,0)  s1=(1,0)
#   s2=(0,1)  s3=(1,1)
#
# Supercell primitive vectors: A1=(2,0), A2=(0,2)
# Reduced BZ: qx,qy in [-pi/2, pi/2)
#
# Spin-orbital ordering in the fragment:
#   [s0_up, s0_dn, s1_up, s1_dn, s2_up, s2_dn, s3_up, s3_dn]
#   (index 2*alpha = up, 2*alpha+1 = dn for site alpha)
#
# The Hamiltonian is split into intra-cluster and inter-cluster parts:
#
#   eloc (intra-cluster, q-independent):
#     h_intra = t * [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]
#     (the 4 NN bonds inside the plaquette)
#
#   ek(q) (inter-cluster, q-dependent):
#     H_{01}(q) = t e^{2iqx}   H_{02}(q) = t e^{2iqy}
#     H_{13}(q) = t e^{2iqy}   H_{23}(q) = t e^{2iqx}
#     (each s_alpha in cell 0 connected to its image in neighboring supercell)
# Full 8x8 ek = kron(H_inter_spinless, eye(2)).
#
# With 1 bath level per impurity orbital (B=1), ntot=16 and the
# (N=8, Sz=0) sector has C(8,4)^2 = 4900 states — fast to diagonalise.
# Increase B for better bath resolution at strong coupling.
# =============================================================

# -------------------------------------------------------
# ANSI colour codes for the D4 symmetry diagnostic
# -------------------------------------------------------
_RESET  = '\033[0m'
_YELLOW = '\033[93m'
_RED    = '\033[91m'

# -------------------------------------------------------
# D4 symmetry enforcement
# -------------------------------------------------------
# D4 = symmetry group of the square, order 8.
# Sites: s0=(0,0), s1=(1,0), s2=(0,1), s3=(1,1) — centre (0.5,0.5)
# Operation   site permutation P (P[alpha] = image of site alpha)
#   E         [0,1,2,3]   identity
#   C4        [1,3,0,2]   90° CCW:  s0->s1->s3->s2->s0
#   C2        [3,2,1,0]   180°:     s0<->s3, s1<->s2
#   C4^3      [2,0,3,1]   270° CCW: s0->s2->s3->s1->s0
#   sigma_x   [2,3,0,1]   reflect y=0.5: s0<->s2, s1<->s3
#   sigma_y   [1,0,3,2]   reflect x=0.5: s0<->s1, s2<->s3
#   sigma_d1  [0,2,1,3]   reflect y=x:   s1<->s2
#   sigma_d2  [3,1,2,0]   reflect y=1-x: s0<->s3
#
# Spin-orbital permutation: index 2*alpha+sigma  ->  2*P[alpha]+sigma
# Bath permutation: bath index b*nimp+i  ->  b*nimp + spin_orb_perm[i]
# Symmetrisation:   M_sym = (1/8) sum_g  M[ inv_p_g, :][:, inv_p_g ]
#   where inv_p_g = argsort(p_g) is the inverse permutation of p_g.

_D4_SITE_PERMS = np.array([
    [0, 1, 2, 3],  # E
    [1, 3, 0, 2],  # C4
    [3, 2, 1, 0],  # C2
    [2, 0, 3, 1],  # C4^3
    [2, 3, 0, 1],  # sigma_x
    [1, 0, 3, 2],  # sigma_y
    [0, 2, 1, 3],  # sigma_d1
    [3, 1, 2, 0],  # sigma_d2
], dtype=int)


def _build_D4_perms(nsite, nimp, nbath):
    """Pre-build all D4 inverse permutations for the impurity and bath spaces."""
    B = nbath // nimp
    imp_inv_perms  = []
    bath_inv_perms = []
    for site_p in _D4_SITE_PERMS:
        # spin-orbital permutation (nimp-dimensional)
        p_imp = np.empty(nimp, dtype=int)
        for alpha in range(nsite):
            for sigma in range(2):
                p_imp[2 * alpha + sigma] = 2 * site_p[alpha] + sigma
        # bath permutation (nbath-dimensional): same site permutation per ghost level
        p_bath = np.empty(nbath, dtype=int)
        for b in range(B):
            p_bath[b * nimp:(b + 1) * nimp] = b * nimp + p_imp
        imp_inv_perms.append(np.argsort(p_imp))
        bath_inv_perms.append(np.argsort(p_bath))
    return imp_inv_perms, bath_inv_perms


def impose_D4_plaquette_symmetry(fragment, nsite=4,
                                 eps_nowarning=1e-5, eps_warning=1e-3,
                                 verbose=True):
    """
    Project Lambda, R, Lambda_c, D onto the D4-invariant subspace by
    averaging over all 8 symmetry operations of the square plaquette.

    Assumes spin-orbital ordering [s0_up, s0_dn, s1_up, s1_dn, ...] and
    bath ordering [ghost_level_0 | ghost_level_1 | ...], each block having
    the same nimp-dimensional site-spin structure.

    Parameters
    ----------
    fragment      : Fragment object (modified in place)
    nsite         : number of plaquette sites (must be 4)
    eps_nowarning : Frobenius norm threshold below which output is uncoloured
    eps_warning   : Frobenius norm threshold above which output is printed in red
    verbose       : if True, print |Delta M|_F for each matrix with colour coding
    """
    nimp  = fragment.nimp
    nbath = fragment.nbath
    assert nimp == nsite * 2, f"Expected nimp={nsite * 2} for {nsite} sites, got {nimp}"

    imp_inv_perms, bath_inv_perms = _build_D4_perms(nsite, nimp, nbath)
    n_ops = len(_D4_SITE_PERMS)

    def _sym(M, row_inv_perms, col_inv_perms):
        out = np.zeros_like(M)
        for ri, ci in zip(row_inv_perms, col_inv_perms):
            out += M[np.ix_(ri, ci)]
        return out / n_ops

    Lambda_sym   = _sym(fragment.Lambda,   bath_inv_perms, bath_inv_perms)
    R_sym        = _sym(fragment.R,        bath_inv_perms, imp_inv_perms)
    Lambda_c_sym = _sym(fragment.Lambda_c, bath_inv_perms, bath_inv_perms)
    D_sym        = _sym(fragment.D,        bath_inv_perms, imp_inv_perms)

    if verbose:
        for name, before, after in [
            ('Lambda',   fragment.Lambda,   Lambda_sym  ),
            ('R',        fragment.R,        R_sym       ),
            ('Lambda_c', fragment.Lambda_c, Lambda_c_sym),
            ('D',        fragment.D,        D_sym       ),
        ]:
            diff = np.linalg.norm(before - after, 'fro')
            msg = f'  D4 |Delta {name}|_F = {diff:.3e}'
            if diff > eps_warning:
                print(f'{_RED}{msg}{_RESET}')
            elif diff > eps_nowarning:
                print(f'{_YELLOW}{msg}{_RESET}')
            else:
                print(msg)

    fragment.Lambda   = Lambda_sym
    fragment.R        = R_sym
    fragment.Lambda_c = Lambda_c_sym
    fragment.D        = D_sym


# -------------------------------------------------------
# Observables
# -------------------------------------------------------

def compute_docc_per_site(fragment, nsite):
    """
    Double occupancy d_alpha = <n_{alpha,up} n_{alpha,dn}> for each impurity site.

    Uses the same thermal averaging as calc_density_matrix, acting on the
    n_i * n_{i+1} operator built by _build_docc_op_sector (orbital i = 2*alpha,
    i+1 = 2*alpha+1 in the full embedding basis).
    """
    solver = fragment.solver
    docc   = np.zeros(nsite)
    for alpha in range(nsite):
        i_up = 2 * alpha   # up-spin orbital index in the embedding basis
        for s in range(len(solver.sectors)):
            bw_s = np.asarray(solver.bw_per_sector[s])
            if len(bw_s) == 0:
                continue
            docc_op_s = solver._build_docc_op_sector(i_up, s)
            U_s       = solver.evecs_list[s]
            docc[alpha] += np.trace(
                U_s.conj().T @ docc_op_s @ U_s @ np.diag(bw_s)).real
    return docc / solver.Zpart


def compute_plaquette_hoppings(fragment, nsite):
    """
    Average NN and NNN elements of the impurity 1-body density matrix.

    NN bonds (4 per spin): (s0-s1), (s0-s2), (s1-s3), (s2-s3)
    NNN bonds (2 per spin): (s0-s3), (s1-s2)

    Spin-orbital index for site alpha, spin sigma: 2*alpha + sigma.
    Returns the real part of the spin-averaged off-diagonal DM elements.
    """
    dm = fragment.denMat[:fragment.nimp, :fragment.nimp].real

    # up-spin index for site alpha = 2*alpha; dn-spin = 2*alpha+1
    nn_pairs_up  = [(0, 2), (0, 4), (2, 6), (4, 6)]
    nn_pairs_dn  = [(1, 3), (1, 5), (3, 7), (5, 7)]
    nnn_pairs_up = [(0, 6), (2, 4)]
    nnn_pairs_dn = [(1, 7), (3, 5)]

    nn_up  = [dm[i, j] for i, j in nn_pairs_up]
    nn_dn  = [dm[i, j] for i, j in nn_pairs_dn]
    nnn_up = [dm[i, j] for i, j in nnn_pairs_up]
    nnn_dn = [dm[i, j] for i, j in nnn_pairs_dn]

    nn_hop  = 0.5 * (np.mean(nn_up)  + np.mean(nn_dn))
    nnn_hop = 0.5 * (np.mean(nnn_up) + np.mean(nnn_dn))
    # also return individual bond values for scatter / symmetry check
    nn_bonds  = 0.5 * (np.array(nn_up)  + np.array(nn_dn))
    nnn_bonds = 0.5 * (np.array(nnn_up) + np.array(nnn_dn))
    return nn_hop, nnn_hop, nn_bonds, nnn_bonds


# =============================================================
# Model and lattice
# =============================================================
t     = 0.25   # nearest-neighbor hopping
B     = 1      # bath levels per impurity spin-orbital
nsite = 4      # sites in the 2x2 plaquette
nimp  = nsite * 2   # spin-orbitals in the fragment: 8
nbath = nimp  * B   # total bath orbitals
ntot  = nimp  + nbath  # embedding problem size: 16

# -------------------------------------------------------
# Build k-space dispersion in the reduced BZ
# -------------------------------------------------------
Nk = 50   # k-points per dimension (Nk^2 total)
qx_list = np.linspace(-np.pi / 2, np.pi / 2, Nk, endpoint=False)
qy_list = np.linspace(-np.pi / 2, np.pi / 2, Nk, endpoint=False)

ek_list = []
for qx in qx_list:
    for qy in qy_list:
        g_x = t * np.exp(2j * qx)   # inter-cluster H_{01} = H_{23}
        g_y = t * np.exp(2j * qy)   # inter-cluster H_{02} = H_{13}
        Hk_spinless = np.array([
            [0.0,        g_x,        g_y,        0.0       ],
            [g_x.conj(), 0.0,        0.0,        g_y       ],
            [g_y.conj(), 0.0,        0.0,        g_x       ],
            [0.0,        g_y.conj(), g_x.conj(), 0.0       ],
        ], dtype=np.complex128)
        ek_list.append(np.kron(Hk_spinless, np.eye(2))) # spin equivalence

eks = np.array(ek_list)
wks = np.ones(len(ek_list)) / len(ek_list)

# Intra-cluster hopping matrix (goes into eloc; q-independent)
h_intra_spinless = t * np.array([[0, 1, 1, 0],
                                  [1, 0, 0, 1],
                                  [1, 0, 0, 1],
                                  [0, 1, 1, 0]], dtype=np.float64)
h_intra = np.kron(h_intra_spinless, np.eye(2))

# Non-interacting kinetic energy (sanity check; full dispersion = ek + h_intra)
ekin0 = 0.0
for ek, wk in zip(eks, wks):
    evals = np.linalg.eigvalsh(ek + h_intra)
    ekin0 += wk * np.sum(evals[evals < 0.0])
print(f'Non-interacting kinetic energy (per plaquette, both spins): {ekin0:.6f}')
print(f'  => per site: {ekin0 / nsite:.6f}')

lattice = Lattice(eks, wk_list=wks)

# =============================================================
# Self-consistency parameters
# =============================================================
itmax    = 50
mix      = 0.3
tol      = 1e-3
T        = 0.0
mu       = 0.0
spin_pen = 0.0   # S^2 penalty to stabilise the paramagnetic singlet sector

# --- symmetry flags ---
impose_site_symmetry = True   # project Lambda/R/Lambda_c/D onto D4-invariant subspace
eps_nowarning        = 1e-5   # |Delta M|_F printed in normal colour below this
eps_warning          = 1e-3   # |Delta M|_F printed in red above this (yellow in between)

# =============================================================
# U scan
# =============================================================
U_list = np.linspace(0.1, 0.5, 5)
Z_list        = []
docc_list     = []    # average docc per site
docc_site_list = []  # (nsite,) array per U — all 4 sites; equal under D4
nn_avg_list   = []   # average NN <c†_i c_j> (spin-averaged)
nnn_avg_list  = []   # average NNN <c†_i c_j> (spin-averaged)
nn_bonds_list = []   # (4,) individual NN bonds per U
nnn_bonds_list= []   # (2,) individual NNN bonds per U

# Initial self-energy: Lambda=0, R ~ 0.8*I (B=1 => square nimp x nimp matrices)
Lambda0 = h_intra#np.zeros((nbath, nbath), dtype=np.complex128)
R0      = np.eye(nbath, dtype=np.complex128)


for iU, U in enumerate(U_list):

    # Local Hamiltonian: intra-cluster hopping + -U/2 diagonal (particle-hole symmetry)
    eloc = h_intra + np.diag(np.full(nimp, -U / 2.0))

    # Hubbard-U on each site: U * n_{alpha,up} * n_{alpha,dn}
    # Convention: Utensor[i,j,k,l] contributes 0.5*U * c†_i c†_k c_l c_j
    Utensor = np.zeros((nimp,) * 4)
    for alpha in range(nsite):
        up, dn = 2 * alpha, 2 * alpha + 1
        Utensor[up, up, dn, dn] = U
        Utensor[dn, dn, up, up] = U

    edsolver = SimpleED(ntot, use_Ntot=True, use_Sz=True,
                        N_sector=ntot // 2, Sz_sector=0,
                        dtype=np.complex128)
    fragment = Fragment(nimp, nbath, eloc, Utensor, edsolver,
                        Lambda=Lambda0, R=R0, verbose=0)

    for it in range(itmax):
        lattice.solve_qp([fragment], T=T)

        # Update hybridization, then enforce all active symmetries
        fragment.update_hybridization(T=T)
        if impose_site_symmetry:
            impose_D4_plaquette_symmetry(fragment, nsite=nsite,
                                         eps_nowarning=eps_nowarning,
                                         eps_warning=eps_warning,
                                         verbose=False)
        fragment.impose_spin_SU2_symmetry()

        fragment.solve_impurity(mu, T=T, num_eig=20, spin_pen=spin_pen)


        Lambda_old = fragment.Lambda.copy()
        R_old      = fragment.R.copy()

        # Update self-energy, then enforce all active symmetries
        fragment.update_self_energy(T=T)
        fragment.impose_spin_SU2_symmetry()

        diff = max(
            np.abs(fragment.Lambda - Lambda_old).max(),
            np.abs(fragment.R      - R_old     ).max(),
        )

        # Mix and re-enforce symmetries; report D4 deviation at end of iteration
        fragment.Lambda = (1 - mix) * fragment.Lambda + mix * Lambda_old
        fragment.R      = (1 - mix) * fragment.R      + mix * R_old
        fragment.impose_spin_SU2_symmetry()
        if impose_site_symmetry:
            impose_D4_plaquette_symmetry(fragment, nsite=nsite,
                                         eps_nowarning=eps_nowarning,
                                         eps_warning=eps_warning,
                                         verbose=True)

        print(f'U={U:.2f}  it={it:3d}  diff={diff:.2e}')

        if diff < tol and it > 3:
            print(f'  --> Converged at it={it}')
            break

    # Warm-start for next U
    Lambda0 = fragment.Lambda.copy()
    R0      = fragment.R.copy()

    # Quasiparticle weight: average Z over the 4 plaquette sites (up-spin diagonal)
    Z          = fragment.compute_Z()
    print(f'  Z full diagonal (all {nimp} spin-orbitals): {[round(Z[i,i].real,6) for i in range(nimp)]}')
    Z_avg      = np.mean([Z[2 * alpha, 2 * alpha].real for alpha in range(nsite)])
    Z_per_site = [round(Z[2 * alpha, 2 * alpha].real, 4) for alpha in range(nsite)]

    dm      = fragment.denMat[:nimp, :nimp].real
    n_total = np.trace(dm)

    # Double occupancy per site
    docc           = compute_docc_per_site(fragment, nsite)
    docc_avg       = float(np.mean(docc))

    # Inter-site hoppings from the impurity 1-body density matrix
    nn_avg, nnn_avg, nn_bonds, nnn_bonds = compute_plaquette_hoppings(fragment, nsite)

    print(f'U={U:.2f}  Z_avg={Z_avg:.4f}  n_tot={n_total:.4f}  Z/site={Z_per_site}')
    print(f'       docc_avg={docc_avg:.4f}  docc/site={np.round(docc, 4).tolist()}')
    print(f'       NN_hop={nn_avg:.4f}  NNN_hop={nnn_avg:.4f}  '
          f'NN_bonds={np.round(nn_bonds, 4).tolist()}  NNN_bonds={np.round(nnn_bonds, 4).tolist()}')

    Z_list.append(Z_avg)
    docc_list.append(docc_avg)
    docc_site_list.append(docc.copy())
    nn_avg_list.append(nn_avg)
    nnn_avg_list.append(nnn_avg)
    nn_bonds_list.append(nn_bonds.copy())
    nnn_bonds_list.append(nnn_bonds.copy())

# =============================================================
# Plots
# =============================================================
docc_site_arr  = np.array(docc_site_list)   # (nU, nsite)
nn_bonds_arr   = np.array(nn_bonds_list)    # (nU, 4)
nnn_bonds_arr  = np.array(nnn_bonds_list)   # (nU, 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Square lattice — 2×2 plaquette fragment (B=1, PM)', fontsize=12)

# --- Panel 1: quasiparticle weight ---
ax = axes[0]
ax.plot(U_list, Z_list, 'o-', color='C0', markersize=4, lw=1.5, label='Z avg')
ax.set_xlabel('U')
ax.set_ylabel('Z')
ax.set_title('Quasiparticle weight')
ax.set_xlim(U_list[0], U_list[-1])
ax.set_ylim(0, 1.05)
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.legend(fontsize=9)

# --- Panel 2: double occupancy per site ---
ax = axes[1]
# individual sites as faint scatter to show D4 quality
for alpha in range(nsite):
    ax.scatter(U_list, docc_site_arr[:, alpha],
               s=12, color='C1', alpha=0.35, zorder=2)
ax.plot(U_list, docc_list, 'o-', color='C1', markersize=4, lw=1.5,
        label=r'$\langle n_\uparrow n_\downarrow \rangle$ avg')
# non-interacting reference d = n_up * n_dn = (1/2)^2 = 0.25 at half-filling
ax.axhline(0.25, color='gray', ls='--', lw=0.8, label='non-int (0.25)')
ax.set_xlabel('U')
ax.set_ylabel(r'$d = \langle n_\uparrow n_\downarrow \rangle$')
ax.set_title('Double occupancy per site')
ax.set_xlim(U_list[0], U_list[-1])
ax.legend(fontsize=9)

# --- Panel 3: inter-site hopping in impurity DM ---
ax = axes[2]
# individual bonds as faint scatter
for b in range(4):
    ax.scatter(U_list, nn_bonds_arr[:, b],
               s=12, color='C2', alpha=0.35, zorder=2)
for b in range(2):
    ax.scatter(U_list, nnn_bonds_arr[:, b],
               s=12, color='C3', alpha=0.35, zorder=2)
ax.plot(U_list, nn_avg_list,  'o-', color='C2', markersize=4, lw=1.5,
        label=r'NN $\langle c^\dagger_i c_j \rangle$ (4 bonds)')
ax.plot(U_list, nnn_avg_list, 's-', color='C3', markersize=4, lw=1.5,
        label=r'NNN $\langle c^\dagger_i c_j \rangle$ (2 bonds)')
ax.axhline(0, color='k', lw=0.5, ls='--')
ax.set_xlabel('U')
ax.set_ylabel(r'$\langle c^\dagger_i c_j \rangle$')
ax.set_title('Impurity inter-site hopping (1-body DM)')
ax.set_xlim(U_list[0], U_list[-1])
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('observables_plaquette_2x2.png', dpi=100)
print('Saved observables_plaquette_2x2.png')
plt.show()
