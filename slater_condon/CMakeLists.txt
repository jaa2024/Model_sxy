cmake_minimum_required(VERSION 3.28)

# 设置项目名称和语言
project(SC_Full_CI LANGUAGES CXX)

# 设置 C++ 标准为 C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 添加额外的编译选项（只在 Release 模式下）
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(UNIX)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    endif()
    if(MSVC)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2")
    endif()
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)

enable_testing()
# 添加可执行文件
add_executable(SC_Full_CI main.cpp)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)


target_link_libraries( SC_Full_CI
        Threads::Threads
        OpenMP::OpenMP_CXX
        )

        