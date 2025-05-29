cmake_minimum_required(VERSION 3.28)

project(IDMRG LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-DEIGEN_USE_THREADS)

# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
# --------------------> Eigen3 <--------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen3)
# --------------------> fmt <--------------------
add_subdirectory(external/fmt)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE src "src/*.[ch]pp")

add_executable(IDMRG main.cpp ${src})
target_link_libraries(IDMRG PRIVATE OpenMP::OpenMP_CXX Threads::Threads fmt::fmt)