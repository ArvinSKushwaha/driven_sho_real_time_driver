cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_ARCHITECTURES all)

project(simulator CXX)

include(FetchContent)
FetchContent_Declare(
	fmt
	GIT_REPOSITORY	https://github.com/fmtlib/fmt.git
    GIT_TAG 	    e69e5f977d458f2650bb346dadf2ad30c5320281 # Release 10.2.1 (04.01.24)
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(fmt)

find_package(fmt)
find_package(OpenMP)

add_subdirectory(src)

target_compile_options(${PROJECT_NAME}
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler> $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic -Werror> -Wall -O3 -fopenmp -fopenmp-simd -Wno-tautological-compare)
target_link_options(${PROJECT_NAME} PUBLIC -fopenmp -fopenmp-simd)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
