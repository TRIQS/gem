.. _user_guide:

User Guide
**********

This section provides a brief introduction to the theory behind gem and
worked examples showing how to set up and run calculations.

.. contents::
   :local:
   :depth: 2

Theoretical Background
======================

A proper presentation of the method has been given in many works for
zero :cite:`Lanata2017` and finite :cite:`Giuli2026` temperature.
Here, we give a brief recap.

The Ghost Gutzwiller Approximation
------------------------------------

The Gutzwiller approximation (GA) is a variational method for lattice models of
strongly correlated electrons with local interaction :cite:`Gutzwiller1965,Bunemann1998`.
The trial state is obtained by applying a linear map :math:`\hat{P} = \prod_i \hat{P}_i`
(often called a projector, despite not being one strictly speaking) to an uncorrelated Slater
determinant :math:`|\Psi_0\rangle`.
Each on-site operator :math:`\hat{P}_i` re-weights the local many-body configurations,
suppressing or enhancing occupancies relative to the non-interacting reference.
This method provided the first explanation for the Mott transition :cite:`Mott1968` via a
renormalization of the hopping parameters thanks to Brinkmann and Rice :cite:`BrinkmanRice1970`.
Many slave-particle mean-field formulations, most notably Kotliar–Ruckenstein and
rotationally invariant slave-boson (RISB) approaches, are equivalent at the saddle-point level
to the multiorbital Gutzwiller approximation, differing mainly in the auxiliary-field representation
of the same variational energy functional :cite:`KotliarRuckenstein1986,Lechermann2007,BuenemannGebhard2007`.

The standard GA is limited because its embedding Hamiltonian is an *interacting*
single-impurity model with at most one bath orbital per correlated orbital.
This restricts the representable spectral structures, missing high-energy features like Hubbard
bands, and prevents a systematic improvement of the approximation toward a dynamical description
of strongly correlated systems as in Dynamical Mean-Field Theory (DMFT) :cite:`Georges1996`.

Ghost-GA :cite:`Lanata2017` extends the embedding Hamiltonian by adding *auxiliary* (ghost) orbitals
to the uncorrelated Slater determinant :math:`|\Psi_0\rangle`, considering a number of electronic
levels that is a multiple :math:`B` of the physical ones.

* At :math:`B = 1` the method reduces to the standard GA.
* As :math:`B \to \infty` it has been proved that the method converges to DMFT :cite:`Giuli2026`.
* For finite :math:`B > 1` it provides a controlled, systematically improvable
  approximation :cite:`Lee2023`.

The variational parameters are the bath hybridisation matrix and the on-site embedding energies.
One of the key advantages of this method is that its self-consistency is rooted in static
expectation values rather than dynamical correlators (as in DMFT), enabling much faster
calculations of correlated matter.

In Ref. :cite:`Giuli2026` it has been shown that Ghost-GA admits a free-energy functional
formulation that connects it with DMFT and allows for a natural finite-temperature extension.
This framework embeds the correlated site in an effective Anderson impurity model at temperature
:math:`T`, and the self-consistency equations remain rooted in static — now thermal — expectation
values.  The method interpolates smoothly between zero-temperature ghost-GA (:math:`T \to 0`)
and conventional DMFT (:math:`B \to \infty`), retaining the low computational cost of ghost-GA
for moderate :math:`B`.

The ghost embedding functional
----------------------

In Ref. :cite:`Giuli2026`, a free-energy functional formulation of the method has been derived, which we briefly recap here.
Consider a general multi-orbital Hubbard Hamiltonian written as

.. math::
   :label: H_split

   \hat H = \hat H_0 + \hat H_{\mathrm{int}} 

where the interaction part is assumed to be local,

.. math::
   :label: Hint_def

   \hat H_{\mathrm{int}}
   =
   \sum_{i=1}^{\mathcal{N}}
   \hat H^{i}_{\mathrm{int}}


and the one-body part is

.. math::
   :label: H0_def

   \hat H_0
   =
   \sum_{i,j=1}^{\mathcal{N}}
   \sum_{\alpha=1}^{\nu_i}
   \sum_{\beta=1}^{\nu_j}
   [h_0]_{i\alpha,j\beta}\,
   c^\dagger_{i\alpha} c_{j\beta}
   \, .

Here :math:`i=1,\ldots, \mathcal{N}` labels system fragments, each hosting
:math:`\nu_i` physical fermionic modes with annihilation operators
:math:`c_{i\alpha}` (:math:`\alpha=1,\ldots,\nu_i`), including both spin
and orbital degrees of freedom.

We represent the one-body matrix :math:`h_0` in block form as

.. math::
   :label: h0_block

   h_0 =
   \begin{pmatrix}
   \epsilon_1 & t_{12} & \dots & t_{1\mathcal{N} } \\
   t_{21} & \epsilon_2 & \dots & \vdots \\
   \vdots & \vdots & \ddots & \vdots \\
   t_{\mathcal{N}1} & \dots & \dots & \epsilon_{\mathcal{N}}
   \end{pmatrix}

where :math:`\epsilon_i \in \mathbb{C}^{\nu_i \times \nu_i}` are Hermitian
local one-body blocks and
:math:`t_{ij} \in \mathbb{C}^{\nu_i \times \nu_j}` (:math:`i \neq j`) are
the hopping blocks satisfying
:math:`t_{ji}=t_{ij}^{\dagger}` as a consequence of the Hermiticity of :math:`h_0`.

We also define the hopping matrix :math:`t` as the block matrix collecting
the off-diagonal blocks of :math:`h_0`:

.. math::
   :label: block_t

   t =
   \begin{pmatrix}
   \mathbf{0} & t_{12} & \dots & t_{1\mathcal{N} } \\
   t_{21} & \mathbf{0} & \dots & \vdots \\
   \vdots & \vdots & \ddots & \vdots \\
   t_{\mathcal{N}1} & \dots & \dots & \mathbf{0}
   \end{pmatrix}
   \, ,

and the corresponding block-diagonal local one-body matrix

.. math::
   :label: eps_def

   \epsilon =
   \begin{pmatrix}
   \epsilon_1 & \mathbf{0} & \dots & \mathbf{0} \\
   \mathbf{0} & \epsilon_2 & \dots & \vdots \\
   \vdots & \vdots & \ddots & \vdots \\
   \mathbf{0} & \dots & \dots & \epsilon_{\mathcal{N}}
   \end{pmatrix},
   \qquad
   h_0 = \epsilon + t \, .

Grouping the local one-body terms and the local interactions into a single local operator

.. math::
   :label: Hloc_def

   \hat H^{i}_{\mathrm{loc}}
   \big[c^\dagger_{i\alpha}, c_{i\alpha}\big]
   =
   \sum_{\alpha,\beta=1}^{\nu_i}
   [\epsilon_i]_{\alpha\beta}\,
   c^\dagger_{i\alpha} c_{i\beta}
   +
   \hat H^{i}_{\mathrm{int}}
   \, ,

the full Hamiltonian at Eq. :eq:`H_split` can be rewritten as:

.. math::
   :label: H_def

   \hat H
   =
   \sum_{\substack{i,j=1 \\ i \neq j}}^{\mathcal{N}}
   \sum_{\alpha=1}^{\nu_i}
   \sum_{\beta=1}^{\nu_j}
   [t_{ij}]_{\alpha\beta}\,
   c^\dagger_{i\alpha} c_{j\beta}
   +
   \sum_{i=1}^{\mathcal{N}}
   \hat H^{i}_{\mathrm{loc}}
   \big[c^\dagger_{i\alpha}, c_{i\alpha}\big]
   \, ,


The method is built on the following dynamical functional:

.. math::
   :label: Lbeta_DMFT

   \begin{align}
   \mathcal{L}_\beta [ {\Sigma_i(i\omega_n)} , {\Delta_i(i\omega_n) }]
   = -& T\sum_{n}e^{i\omega_n0^+} \text{Tr}
   \ln \big(i \omega_n \mathbf{1}-h_0-\Sigma(i \omega_n ) \big)
   \nonumber \\
   +& T\sum_{i=1}^{\mathcal{N}} \sum_{n}e^{i\omega_n0^+}\text{Tr}
   \ln \big(i \omega_n \mathbf{1} -  \epsilon_i - \Delta_i(i \omega_n ) -  \Sigma_i(i \omega_n )\big)
   \nonumber \\
   +& \sum_{i=1}^{\mathcal{N}} \Omega_{\rm imp}^i[\Delta_i(i \omega_n )]
   \end{align}

Where :math:`\Sigma(i\omega_n)= \text{diag}(\Sigma_1(i\omega_n), \ldots, \Sigma_{\mathcal{N}}(i\omega_n))`
The saddle point equations of this functional are the DMFT self-consistency conditions.
Introducing the following parametrization of the self-energy and hybridization functions:

.. math::
   :label: Sigma_Delta_param

   \begin{align}
   \Sigma_i(z) &=
   z\mathbf{1}_{\nu_i} -\epsilon_i
   -\Big[  \mathcal{R}_i^\dagger \big( z\mathbf{1}_{B\nu_i}-\Lambda_i\big)^{-1}\mathcal{R}_i \Big]^{-1}
   \\
   \Delta_i(z) &=
   \mathcal{D}_i^{T}\big(z\mathbf{1}+\Lambda_i^c\big)^{-1}\mathcal{D}_i^{*}
   \end{align}

where :math:`\mathcal{R}_i, \mathcal{D}_i` are complex matrices of shape :math:`B \nu_i \times \nu_i`
and :math:`\Lambda_i, \Lambda_i^c` are complex hermitian matrices, the functional can be rewritten as:

.. math::
   :label: Lbeta_def_req

   \begin{align}
   \mathcal{L}_\beta[\mathcal{R},\Lambda,\mathcal{D},\Lambda^c]
   =&
   \Omega_{\rm qp}[\mathcal{R},\Lambda]
   \nonumber \\
   -&\sum_{i=1}^{\mathcal{N}}\Omega_{0,{\rm emb}}^i[\mathcal{D}_i,\Lambda_i^c;\mathcal{R}_i,\Lambda_i]
   \nonumber \\
   +&\sum_{i=1}^{\mathcal{N}}\Omega_{\rm emb}^i[\mathcal{D}_i,\Lambda_i^c]
   \end{align}

where :math:`\Omega_{\alpha}` are the free-energies associated to the following Hamiltonian:

.. math::
   :label: H_emb

   \begin{align}
   \hat{H}_{\text{qp}}        &= \sum_{ij} \sum_{a=1}^{\mathcal{B} \nu_i} \sum_{b=1}^{\mathcal{B} \nu_j} f^\dagger_{i a} \, [ \delta_{ij} (\Lambda^i)_{ ab} + \sum_{\alpha \beta} R_{i, a \alpha} \, t^{ij}_{\alpha\beta} \, R^\dagger_{j,\beta b}  ] \, f_{ j b}\\
   \hat{H}^{i}_{\text{emb}}     &= \hat{H}^{i}_{\text{int}}[ c^\dagger_i ,c_i ] + \sum_{ a=1}^{\mathcal{B} \nu_i} \sum_{ \alpha=1 }^{\nu_i} [ (D^i)_{ a \alpha } \, b^\dagger_{i, a} c_{i, \alpha} + \text{h.c.} ]
                          + \sum_{\alpha\beta=1}^{\mathcal{B} \nu_i} (\Lambda_c^i)_{\alpha\beta} \, f^\dagger_\alpha b_\beta \\
   \hat{H}^{i}_{0,\text{emb}} &= \sum_{a,b=1}^{B\nu_i} \big[\Lambda^i\big]_{ab}\, f^\dagger_{ia} f_{ib} +  \sum_{a,b=1}^{B \nu_i}  \Big(\big[D^i R_i^{T}\big]_{ab}\, f^\dagger_{ib} b_{ia}+\text{H.c.}\Big)+ \sum_{a,b=1}^{B\nu_i}\big[\Lambda^i_c\big]_{ab}\,b^\dagger_{ib} b_{ia}
   \end{align}

The saddle point equations of the functional at :math:`T=0` retrieve the ghost-GA self-consistency equations and generalize them to finite temperature, while the limit :math:`B \to \infty` retrieves the DMFT functional and self-consistency equations.



Self-Consistency Loop
----------------------

The self-consistency loop of the method is based on the saddle point equations of the functional at Eq. :eq:`Lbeta_def_req`, which are given in the following:

.. math::
   :label: saddle_R

   \frac{\partial \mathcal{L}_\beta}{\partial [\mathcal{R}_i]_{a \alpha} } = 0 \Rightarrow \sum_{j=1}^{\mathcal{N}} \sum_{\beta=1}^{\nu_j} \sum_{b=1}^{B \nu_j} [t^{ij}]_{\alpha \beta} [\mathcal{R}^\dagger_j ]_{\beta b} \langle f^\dagger_{ia} f_{j b} \rangle_{\text{qp}} = \sum_{b=1}^{B \nu_i} [\mathcal{D}_i]_{b \alpha} \langle f^\dagger_{i a} b_{i b} \rangle_{\text{0,emb,i}}

.. math::
   :label: saddle_lambda

   \frac{\partial \mathcal{L}_\beta}{\partial [\Lambda_i]_{a b}} = 0 \Rightarrow \langle f^\dagger_\alpha f_\beta \rangle_{\text{qp}} = \langle f^\dagger_\alpha f_\beta \rangle_{\text{0,emb,i}}

.. math::
   :label: saddle_D

   \frac{\partial \mathcal{L}_\beta}{\partial [\mathcal{D}_i]_{a \alpha} } = 0 \Rightarrow \langle c^\dagger_{i \alpha} b_{ib} \rangle_{\text{emb,i}} =  \langle \Big( \sum_{a=1}^{B\nu_i}[ \mathcal{R}_i ]_{a \alpha} f^\dagger_{ia} \Big)b_{ib} \rangle_{\text{0,emb,i}}

.. math::
   :label: saddle_lambda_c

   \frac{\partial \mathcal{L}_\beta}{\partial [\Lambda_i^c]_{a b} } = 0 \Rightarrow \langle b_\alpha b^\dagger_\beta \rangle_{\text{emb,i}} = \langle b_\alpha b^\dagger_\beta \rangle_{\text{0,emb,i}}



A typical ghost-GA calculation for a single correlated fragment proceeds as follows:

1. **Build the dispersion** — construct the array ``eks`` of shape
   ``(n_k, nimp, nimp)`` and the corresponding k-point weights ``wks``.
2. **Create the lattice** — ``Lattice(eks, wk_list=wks)``.
3. **Choose a solver and create a fragment** —
   ``Fragment(nimp, nbath, eloc, Utensor, solver)``.
4. **Iterate until convergence**:

   a. ``lattice.solve_qp([fragment], T=T)`` — solve the quasiparticle problem and update the quasiparticle target expectation values.
   b. ``fragment.update_hybridization(T=T)`` — Update the parameters :math:`D` and :math:`\Lambda_c` of the hybridization function via the saddle point equations :eq:`saddle_D` and :eq:`saddle_lambda_c`.
   c. ``fragment.solve_impurity(mu, T=T)`` — solve the embedding Hamiltonian and update the impurity target expectation values.
   d. ``fragment.update_self_energy(T=T)`` — Update the parameters  :math:`R` and :math:`\Lambda` of the self-rnergy via the saddle point equations :eq:`saddle_R` and :eq:`saddle_lambda`.
      parameters :math:`R` and :math:`\Lambda` from the density matrix.
   e. Mix old and new parameters and check convergence.

5. **Extract observables** — ``fragment.compute_Z()`` for the quasiparticle
   weight, ``fragment.denMat`` for the impurity density matrix.





Examples
========

.. toctree::
   :maxdepth: 1

   examples/bethe_1orb
   examples/square_afm
   examples/square_phase

