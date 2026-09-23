# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests
  test_Basic
  test_density_fitting
  test_temperature
  test_1o3_simple_ed
  test_2o2_simple_ed
  # test_2o6_simple_ed
  test_symmetries_simple_ed
  test_matrix_free_simple_ed
  test_delta_fit_LR_B3
  test_delta_fit_LR_random
  test_delta_fit_LcD_B3
  test_delta_fit_LcD_random
  test_delta_fit_gradient
  test_delta_fit_packing
  test_lattice_mpi
  test_save_load_fragment
  test_save_load_lattice
)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH})
endforeach()

# Tests that must run on several ranks. ctest launches them itself, one entry
# per rank count, and PROCESSORS tells `ctest -j` how many slots each one takes
# so that it does not oversubscribe the machine.
set(mpi_tests
  test_mpi_fragment
  test_mpi_lattice
)
set(mpi_nprocs 2 4)

if(GEM_MPI_TESTS)
  foreach(test ${mpi_tests})
    foreach(np ${mpi_nprocs})
      add_test(NAME Py_${test}_np${np}
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                       ${MPIEXEC_PREFLAGS}
                       ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.py
                       ${MPIEXEC_POSTFLAGS}
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      set_property(TEST Py_${test}_np${np} APPEND PROPERTY
                   ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH})
      set_tests_properties(Py_${test}_np${np} PROPERTIES PROCESSORS ${np})
    endforeach()
  endforeach()
endif()
