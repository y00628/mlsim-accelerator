cmake_minimum_required(VERSION 3.16)
project(mlsim_golden CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags — important for meaningful benchmarks
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(gemm gemm.cpp)

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm gemm)

add_executable(roofline roofline.cpp)
target_link_libraries(roofline gemm)
