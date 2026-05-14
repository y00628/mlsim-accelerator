cmake_minimum_required(VERSION 3.16)
project(mlsim_systemc CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json so clangd/IntelliSense sees SystemC headers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Find SystemC
#
# Priority order:
#   1. SYSTEMC_HOME env var  (point at your build/install prefix)
#   2. find_package(SystemCLanguage)  (works if SystemC was installed with CMake)
#
# Install SystemC from source (no Homebrew formula):
#   wget https://www.accellera.org/images/downloads/standards/systemc/systemc-3.0.0.tar.gz
#   tar xf systemc-3.0.0.tar.gz && cd systemc-3.0.0
#   mkdir build && cd build
#   cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/systemc
#   make -j$(nproc) && make install
#   export SYSTEMC_HOME=$HOME/systemc
# ---------------------------------------------------------------------------
if(DEFINED ENV{SYSTEMC_HOME})
  set(SYSTEMC_HOME "$ENV{SYSTEMC_HOME}")
  message(STATUS "Using SYSTEMC_HOME = ${SYSTEMC_HOME}")

  find_library(SYSTEMC_LIB
    NAMES systemc
    HINTS
      "${SYSTEMC_HOME}/lib"
      "${SYSTEMC_HOME}/lib-macosx64"
      "${SYSTEMC_HOME}/lib-linux64"
    REQUIRED
  )
  set(SYSTEMC_INCLUDE "${SYSTEMC_HOME}/include")

  add_library(SystemC::systemc UNKNOWN IMPORTED)
  set_target_properties(SystemC::systemc PROPERTIES
    IMPORTED_LOCATION "${SYSTEMC_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${SYSTEMC_INCLUDE}"
  )
else()
  find_package(SystemCLanguage CONFIG QUIET)
  if(NOT SystemCLanguage_FOUND)
    message(FATAL_ERROR
      "SystemC not found.\n"
      "  Option A: set SYSTEMC_HOME to your SystemC install prefix, e.g.:\n"
      "            export SYSTEMC_HOME=\$HOME/systemc\n"
      "  Option B: install SystemC with CMake support and re-run cmake.\n"
      "  See README.md for full install steps.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Golden reference library (used by testbench for correctness checking)
#
# When built via the top-level CMakeLists the `gemm` target already exists;
# reuse it.  When built standalone (cd systemc && cmake ..) build from source.
# ---------------------------------------------------------------------------
if(TARGET gemm)
  set(GEMM_TARGET gemm)
else()
  add_library(gemm_ref STATIC ../golden/gemm.cpp)
  target_include_directories(gemm_ref PUBLIC ../golden)
  set(GEMM_TARGET gemm_ref)
endif()

# ---------------------------------------------------------------------------
# .sc sources — CMake doesn't know this extension; declare them as C++
# ---------------------------------------------------------------------------
set(SC_SOURCES tb_array.sc)
set_source_files_properties(${SC_SOURCES} PROPERTIES LANGUAGE CXX)

# ---------------------------------------------------------------------------
# Testbench executable
# ---------------------------------------------------------------------------
add_executable(tb_array ${SC_SOURCES})
target_include_directories(tb_array PRIVATE .)
target_link_libraries(tb_array PRIVATE SystemC::systemc ${GEMM_TARGET})

# Suppress noisy deprecation warnings from SystemC headers on recent compilers
target_compile_options(tb_array PRIVATE -Wno-deprecated-declarations)
