
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 20)
project(atrous)

include(conan-build/conanbuildinfo.cmake)
conan_basic_setup()

include_directories(extern)
add_subdirectory(src)