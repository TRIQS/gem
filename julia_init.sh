# module purge
# module load triqs/3.2.x_nix2.2_llvm julia
###your julia binary
export PYTHON_JULIAPKG_EXE="/mnt/sw/nix/store/nypzid84k10gh1s6f68f555lka4l18k1-julia-1.9.3/bin/julia"
###the julia folder in gem
export PYTHON_JULIAPKG_PROJECT="/mnt/home/ogingras/Work/Libs_2.3-20240529/TRIQS/gem/julia"
##your python venv
export JULIA_CONDAPKG_BACKEND="Null"
export JULIA_PYTHONCALL_EXE="/mnt/home/ogingras/.py_3.11.7_2.3-20240529/bin/python"
# source /mnt/home/bkloss/projects/gem/.triqs/bin/activate
###Threading layer defaults to sequential after sourcing these modules, so we set it to INTEL manually --- may be breaking things elsewhere!
export MKL_THREADING_LAYER="INTEL"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes

export MKL_NUM_THREADS=3
export JULIA_NUM_THREADS=1

export JULIA_LOAD_PATH=$JULIA_LOAD_PATH:$PYTHON_JULIAPKG_PROJECT/src

# export JULIA_DEPOT_PATH="/mnt/home/bkloss/.julia/v1.9:/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/local/share/julia:/mnt/sw/nix/store/fms6zspq4gnq3x8bps8i9wx4lhd6j8s3-julia-1.9.0/share/julia"
