# module purge
# module load triqs/3.2.x_nix2.2_llvm julia
###your julia binary
export PYTHON_JULIAPKG_EXE="$(which julia)" #"/home/samuele/julia-1.9.3/bin/julia"
###the julia folder in ghostGA
export PYTHON_JULIAPKG_PROJECT="$(pwd)/julia" #"/home/samuele/TRIQS/ghostGA/julia"
##your python venv
export JULIA_CONDAPKG_BACKEND="Null"
export JULIA_PYTHONCALL_EXE="$(which python3)" #"/usr/bin/python3"
# source /mnt/home/bkloss/projects/ghostGA/.triqs/bin/activate
###Threading layer defaults to sequential after sourcing these modules, so we set it to INTEL manually --- may be breaking things elsewhere!
export MKL_THREADING_LAYER="INTEL"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

export MKL_NUM_THREADS=3
export JULIA_NUM_THREADS=1

# export JULIA_DEPOT_PATH="/mnt/home/bkloss/.julia/v1.9:/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/local/share/julia:/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/share/julia"
