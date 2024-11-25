cmake_minimum_required(VERSION 3.28)
include(ExternalProject)

project(CMATRIX LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------> OpenMP <--------------------
option(ENABLE_OPENMP "Use OpenMP parallelization." ON)

if(${ENABLE_OPENMP})
  message("-> Enable OpenMP")
  find_package(OpenMP)
  message("-> OpenMP library: " "${OpenMP_C_LIB_NAMES}")
else()
  message("-> Disable OpenMP")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(UNIX)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    endif()
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2")
    endif()
endif()

# --------------------> intel Math Kernel Library (required) <--------------------
option(USE_MKL "Use Intel MKL (default ON). If OFF, use OpenBLAS." ON)

if(USE_MKL)
  add_definitions(-D__USE_MKL__)
  if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "Unable to locate: intel Math Kernel Library.")
  endif()

  # MKL includes
  include_directories($ENV{MKLROOT}/include)

  # set linking type: dynamic (default) or static
  if(NOT DEFINED MKL_LINKING_TYPE OR "${MKL_LINKING_TYPE}" STREQUAL "")
    set(MKL_LINKING_TYPE "Dynamic" CACHE STRING
      "MKL linking type, options are: Dynamic (default), Static."
      FORCE)
  endif()

  set_property(CACHE MKL_LINKING_TYPE PROPERTY STRINGS Dynamic Static)
  message("-> MKL linking type: " ${MKL_LINKING_TYPE})

  # set threading type
  if(NOT DEFINED MKL_THREADING_LAYER OR "${MKL_THREADING_LAYER}" STREQUAL "")
    set(MKL_THREADING_LAYER "Sequential" CACHE STRING
      "Threading layer of MKL, options are: Sequential (default), OpenMP, TBB."
      FORCE)
  endif()

  set_property(CACHE MKL_THREADING_LAYER PROPERTY STRINGS Sequential OpenMP TBB)
  message("-> MKL threading layer: " ${MKL_THREADING_LAYER})

  # MKL link options
  # ---------------> static linking <---------------
  if(${MKL_LINKING_TYPE} STREQUAL "Static")
    # ----------> MKL with OpenMP <----------
    if(${MKL_THREADING_LAYER} STREQUAL "OpenMP")
      # if use intel classical compiler and iomp5
      if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(MKLLIB "-Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a \
  $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl")

      # if use gcc/g++ and gomp
      elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set(MKLLIB "-Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
  $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
      endif()

    # ----------> MKL with TBB <----------
    elseif(${MKL_THREADING_LAYER} STREQUAL "TBB")
      set(MKLLIB "-Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_tbb_thread.a \
  $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{TBBROOT}/lib/intel64/gcc4.8 -ltbb -lstdc++ -lpthread -lm -ldl")

    # ----------> default, sequential MKL <----------
    else()
      set(MKLLIB "-Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_sequential.a \
  $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl")
    endif()

  # ---------------> dynamic linking <---------------
  else()
    # ----------> default, sequential MKL <----------
    if(${MKL_THREADING_LAYER} STREQUAL "Sequential")
      # if use intel classical compiler
      if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(MKLLIB "-L$ENV{MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")

      # if use gcc/g++
      elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set(MKLLIB "-L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl")
      endif()

    # ----------> parallel MKL (OpenMP/TBB) <----------
    else()
      # if use intel classical compiler and iomp5
      if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        set(MKLLIB "-L$ENV{MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")

      # if use gcc/g++ and gomp
      elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        set(MKLLIB "-L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
      endif()
    endif()
  endif()

  message("-> MKL link line: " ${MKLLIB})

  set(MATHLIBS ${MKLLIB})
  # --------------------> OpenBLAS <--------------------
else()
  ExternalProject_Add(
    openblas
    PREFIX ${CMAKE_BINARY_DIR}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/OpenBLAS-0.3.28
    INSTALL_DIR ${CMAKE_BINARY_DIR}/openblas
    CMAKE_ARGS 
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/openblas
      -DBUILD_SHARED_LIBS=OFF  
      -DDYNAMIC_ARCH=ON
      -DTARGET=GENERIC
      -DUSE_LAPACK=ON
      -DUSE_THREAD=OFF
      -DUSE_OPENMP=OFF
      -DCMAKE_C_FLAGS="-O3"
  )

  ExternalProject_Get_Property(openblas BINARY_DIR)
  get_filename_component(OPENBLASROOT ${BINARY_DIR} ABSOLUTE)

  include_directories(${CMAKE_BINARY_DIR}/openblas/include/openblas/)

  set(MATHLIBS ${OPENBLASROOT}/lib/libopenblas.a)
  message("-> Using OpenBLAS from external directory. Link library: " ${MATHLIBS})
endif()

# --------------------> libcint (required) <--------------------
ExternalProject_Add(
  cint
  PREFIX ${CMAKE_BINARY_DIR}
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/libcint
  INSTALL_DIR ${CMAKE_BINARY_DIR}/libcint
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DENABLE_STATIC=ON
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


# include path
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(CMATRIX src/main.cpp src/cmatrix.cpp)

add_dependencies(${PROJECT_NAME} cint)

if(${ENABLE_OPENMP} AND OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX ${MATHLIBS} ${CINTLIB} ${QUADMATH_LIB} )
else()
  target_link_libraries(${PROJECT_NAME} PUBLIC ${MATHLIBS} ${CINTLIB} ${QUADMATH_LIB} )
endif()