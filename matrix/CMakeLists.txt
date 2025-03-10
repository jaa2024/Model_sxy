cmake_minimum_required(VERSION 3.28)

project(MatrixProject LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# --------------------> MKL (required) <--------------------
message(STATUS "MKL support enabled.")
set(MKLLIB "-L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
include_directories(${MKLROOT}/include)

# --------------------> libcint <--------------------
add_subdirectory(external/libcint)
include_directories(${CMAKE_BINARY_DIR}/external/libcint/include)
# --------------------> fmt <--------------------
add_subdirectory(external/fmt)

file(GLOB_RECURSE src_lianlg "src/linalg/*.[ch]pp")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_executable(Edgerunner ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${src_lianlg})

target_link_libraries(Edgerunner PRIVATE
                      ${MKLLIB}
                      fmt::fmt
                      cint
                      OpenMP::OpenMP_CXX 
                      Threads::Threads  )