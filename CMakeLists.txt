# ##############################################################################
#
# gem - Ghost Embedding Method
#
# A Python package for quantum embedding calculations based on the ghost Gutzwiller approximation.
#
# Copyright (C) 2026, The Simons Foundation
#
#
# gem is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# gem is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# gem (in the file COPYING.txt in this directory). If not, see
# <http://www.gnu.org/licenses/>.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(VERSION 3.20)

# ############
# Define Project
project(gem VERSION 0.1.0 LANGUAGES C CXX)
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# ############
# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(Python_EXECUTABLE ${Python3_EXECUTABLE})

# Get the git hash
execute_process(
  COMMAND git -C ${PROJECT_SOURCE_DIR} rev-parse --short HEAD
  OUTPUT_VARIABLE PROJECT_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT PROJECT_GIT_HASH)
  set(PROJECT_GIT_HASH "unknown")
endif()
message(STATUS "${PROJECT_NAME} version : ${PROJECT_VERSION}")
message(STATUS "${PROJECT_NAME} Git hash: ${PROJECT_GIT_HASH}")

# Compute Python install destination relative to prefix
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import sys, os, sysconfig; print(os.path.relpath(sysconfig.get_path('purelib'), sys.prefix))"
  OUTPUT_VARIABLE PYTHON_LIB_DEST_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Default install directory to Python prefix if not given or invalid.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE _python_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "No install prefix given (or invalid). Defaulting to Python prefix: ${_python_prefix}")
  set(CMAKE_INSTALL_PREFIX ${_python_prefix} CACHE PATH "default install path" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
endif()
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")
endif()
set(${PROJECT_NAME}_BINARY_DIR ${PROJECT_BINARY_DIR} CACHE STRING "Binary directory of the ${PROJECT_NAME} Project")

# Make additional Find Modules available
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/share/cmake/Modules)

# ############
# CMake Options

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")
endif()

# Documentation
option(Build_Documentation "Build documentation" OFF)

# Testing
option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
endif()

# #############
# Build Project

# Find / Build dependencies
add_subdirectory(deps)

# Tests
if(Build_Tests)
  add_subdirectory(test)
endif()

# Python
add_subdirectory(python/${PROJECT_NAME})

# Docs
if(NOT IS_SUBPROJECT AND Build_Documentation)
  add_subdirectory(doc)
endif()

# Additional configuration files
add_subdirectory(share)

# check optional python packages
message(STATUS "------- Optional python packages -------")
function(check_option_python command_str name)
  execute_process(COMMAND ${Python_EXECUTABLE} -c ${command_str}
    OUTPUT_VARIABLE res RESULT_VARIABLE returncode OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT returncode EQUAL 0)
    message(STATUS "-- ${name} is NOT installed.")
    set(FORKTPS_INSTALLED 1 PARENT_SCOPE)
  else()
    message(STATUS "-- ${name} is installed.")
  endif()
endfunction()

check_option_python("import block2" "Block2")

# #############
# Debian Package

option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE AND NOT IS_SUBPROJECT)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/${PROJECT_NAME}")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CMAKE_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "python3, python3-numpy, h5py")
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()
