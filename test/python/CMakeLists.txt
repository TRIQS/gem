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
  test_1o3_simpled_ed
  test_2o6_simple_ed
  test_symmetries_simple_ed
  test_delta_fit_LR_B3
  test_delta_fit_LR_random
  test_delta_fit_LcD_B3
  test_delta_fit_LcD_random
  test_delta_fit_gradient
  test_delta_fit_packing
)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH})
endforeach()
