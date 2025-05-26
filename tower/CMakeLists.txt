cmake_minimum_required(VERSION 3.28)
include(ExternalProject)

# --------------------> Project Configuration <--------------------
project(Tower LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------> Build Mode Selection <--------------------
set(BUILD_TYPE "PYTHON_LIBRARY" CACHE STRING "Build type (EXECUTABLE or PYTHON_LIBRARY)")
set_property(CACHE BUILD_TYPE PROPERTY STRINGS "EXECUTABLE" "PYTHON_LIBRARY")
message(STATUS "Build type: ${BUILD_TYPE}")

# --------------------> Python Configuration <--------------------
if(BUILD_TYPE STREQUAL "PYTHON_LIBRARY")
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# --------------------> Compiler Configuration <--------------------
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")

# Common flags setup
set(COMMON_CXX_FLAGS "-m64")
set(COMMON_C_FLAGS "-m64")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------> Optimization Flags <--------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Applying Release-mode optimizations")
    set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -O3")
    
    # Compiler-specific optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -ipo -xHost")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -march=native --param=destructive-interference-size=64")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -march=native -Wno-unused-command-line-argument")
    endif()
else()
    message(STATUS "Applying non-Release build flags")
    set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -O0 -g")
    set(COMMON_C_FLAGS "${COMMON_C_FLAGS} -O0 -g")
endif()

# Apply global flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_C_FLAGS}")
# --------------------> External Dependencies <--------------------
ExternalProject_Add(
  fmt
  PREFIX ${CMAKE_BINARY_DIR}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/fmt
  INSTALL_DIR ${CMAKE_BINARY_DIR}/fmt
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/fmt
  -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE
  -DFMT_TEST=OFF
  -DFMT_DOC=OFF
)
ExternalProject_Get_Property(fmt BINARY_DIR)
get_filename_component(FMTROOT ${BINARY_DIR} ABSOLUTE)
include_directories(${CMAKE_BINARY_DIR}/fmt/include)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(FMTLIB ${FMTROOT}/libfmt.a)
else()
    set(FMTLIB ${FMTROOT}/libfmtd.a)
endif()
# --------------------> libcint (required) <--------------------
ExternalProject_Add(
  cint
  PREFIX ${CMAKE_BINARY_DIR}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/libcint
  INSTALL_DIR ${CMAKE_BINARY_DIR}/libcint
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DENABLE_STATIC=ON
  -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/libcint
)
ExternalProject_Get_Property(cint BINARY_DIR)
get_filename_component(CINTROOT ${BINARY_DIR} ABSOLUTE)
include_directories(${CINTROOT}/include)
set(CINTLIB "-Wl,--start-group ${CINTROOT}/libcint.a -Wl,--end-group")
execute_process(COMMAND ${CMAKE_C_COMPILER} -print-file-name=libquadmath.a
  OUTPUT_VARIABLE QUADMATH_LIB
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-> Using quadmath library: ${QUADMATH_LIB}")
# --------------------> Core Library (Shared) <--------------------
file(GLOB_RECURSE SOURCES_C "src/integral/*.c")
set_source_files_properties(${SOURCES_C} PROPERTIES LANGUAGE C)
add_library(tower_c_objects OBJECT ${SOURCES_C})
target_include_directories(tower_c_objects PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/eigen
    ${CMAKE_SOURCE_DIR}/external/WignerSymbol
    ${CMAKE_SOURCE_DIR}/src
)
add_library(tower_core INTERFACE)
target_sources(tower_core INTERFACE  $<TARGET_OBJECTS:tower_c_objects>  # 链接 OBJECT 文件
${src_integral}                   # 头文件
)
target_link_libraries(tower_core INTERFACE ${FMTLIB} ${CINTLIB} ${QUADMATH_LIB} -static-libstdc++)
target_compile_definitions(tower_core INTERFACE ${CMAKE_BINA_DIR}
)

# --------------------> BLAS/LAPACK Configuration <--------------------
set(BLAS_TYPE "IntelMKL" CACHE STRING "BLAS implementation")
set_property(CACHE BLAS_TYPE PROPERTY STRINGS "OpenBLAS" "IntelMKL")

if(BLAS_TYPE STREQUAL "OpenBLAS")
    find_package(OpenBLAS REQUIRED)
    target_include_directories(tower_core INTERFACE  ${OpenBLAS_INCLUDE_DIR})
    target_compile_definitions(tower_core INTERFACE  EIGEN_USE_BLAS)
    target_link_libraries(tower_core INTERFACE ${OpenBLAS_LIBRARY})

elseif(BLAS_TYPE STREQUAL "IntelMKL")
    if(NOT DEFINED ENV{MKLROOT})
        message(FATAL_ERROR "MKLROOT environment variable not set! Please source Intel oneAPI environment first.")
    endif()
    set(MKLROOT $ENV{MKLROOT})
    
    target_include_directories(tower_core INTERFACE "${MKLROOT}/include")
    target_compile_definitions(tower_core INTERFACE EIGEN_USE_MKL_ALL)
    
    # MKL linking configuration
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        target_link_libraries(tower_core INTERFACE 
            -L${MKLROOT}/lib 
            mkl_intel_lp64 mkl_intel_thread mkl_core iomp5
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(tower_core INTERFACE 
            -L${MKLROOT}/lib 
            mkl_intel_lp64 mkl_gnu_thread mkl_core gomp
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_link_libraries(tower_core INTERFACE 
            -L${MKLROOT}/lib 
            mkl_intel_lp64 mkl_intel_thread mkl_core iomp5
        )
    endif()
    
    target_link_libraries(tower_core INTERFACE pthread m dl)
endif()

# --------------------> Build Targets <--------------------
if(BUILD_TYPE STREQUAL "EXECUTABLE")
    add_executable(Tower 
    ${CMAKE_SOURCE_DIR}/main.cpp)
    target_link_libraries(Tower PRIVATE tower_core)
    target_include_directories(Tower PRIVATE 
    tower_core
    ${CMAKE_SOURCE_DIR}/external/eigen
    ${CMAKE_SOURCE_DIR}/external/WignerSymbol)
    message(STATUS "Building standalone executable")

elseif(BUILD_TYPE STREQUAL "PYTHON_LIBRARY")
    pybind11_add_module(Tower
        # ${CMAKE_SOURCE_DIR}/pybind11/pybind11_eigen.cpp
        ${CMAKE_SOURCE_DIR}/src/pybind11/pybind11_integral.cpp
        ${CMAKE_SOURCE_DIR}/src/pybind11/pybind11_x2camf.cpp
        ${CMAKE_SOURCE_DIR}/src/pybind11/pybind11.cpp
        # add more
    )
    target_link_libraries(Tower PRIVATE tower_core )
    target_include_directories(Tower PRIVATE 
    tower_core
    ${CMAKE_SOURCE_DIR}/external/eigen 
    ${CMAKE_SOURCE_DIR}/external/WignerSymbol)
    install(TARGETS Tower LIBRARY DESTINATION ${Python_SITEARCH})
    message(STATUS "Building Python library for version ${Python_VERSION}")
endif()