cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20)

project(simulator)

include(FetchContent)
FetchContent_Declare(
	fmt
	GIT_REPOSITORY	https://github.com/fmtlib/fmt.git
    GIT_TAG 	    e69e5f977d458f2650bb346dadf2ad30c5320281 # Release 10.2.1 (04.01.24)
    OVERRIDE_FIND_PACKAGE
    GIT_PROGRESS TRUE
)

find_package(fmt)

add_subdirectory(src)

target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wpedantic -Werror)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
