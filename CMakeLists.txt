cmake_minimum_required(VERSION 3.4...3.18)
project(pysiclib LANGUAGES CXX)

# set(CMAKE_OSX_ARCHITECTURES x86_64)
set(CMAKE_CXX_STANDARD 20)
# set(Python_ROOT_DIR "/opt/homebrew/anaconda3/bin/python")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/..)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB numerical_src CONFIGURE_DEPENDS
	"siclib/numerical/*.cpp"
	"siclib/numerical/*.hpp"
)
file(GLOB linalg_src CONFIGURE_DEPENDS
	"siclib/linalg/*.cpp"
	"siclib/linalg/*.hpp"
)

pybind11_add_module(
	_pysiclib siclib/_siclib_py_bindings.cpp ${numerical_src} ${linalg_src})
