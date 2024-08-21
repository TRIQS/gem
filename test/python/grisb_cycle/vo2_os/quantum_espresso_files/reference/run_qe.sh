
# run the scf calculation
mpirun -n 4 pw.x -i vo2.scf.in | tee vo2.scf.out


# OPTIONAL: run the DFT bandstructure calculation
mpirun -n 4 pw.x -i vo2.bnd.in | tee vo2.bnd.out
mpirun -n 4 bands.x -i vo2.bands.in | tee vo2.bands.out


# run the nscf calculation
mpirun -n 4 pw.x -i vo2.nscf.in | tee vo2.nscf.out

# Wannierize

# pre-process wannier90
wannier90.x -pp vo2
# run interface wannier90-quantum espresso
mpirun -n 4 pw2wannier90.x -i vo2.pw2wan.in | tee vo2.pw2wan.out
# run wannier90
wannier90.x vo2

# Interface with TRIQS, this will create the vo2.h5 archive to start the calculation
#python3 ./convert_wannier.py
