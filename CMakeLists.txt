cmake_minimum_required(VERSION 3.18)
project(pysiclib LANGUAGES CXX)
enable_testing()

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Og -fasynchronous-unwind-tables -D_GLIBCXX_ASSERTIONS -fsanitize-recover=address -fsanitize=address -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=unreachable -fsanitize=null -fsanitize=bounds -fsanitize=alignment"
)

set(CMAKE_CXX_FLAGS_RELEASE "-O2" )

if(APPLE)
	find_library(CPU_BLAS Accelerate)
else()
	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)
endif()



# set(Python_ROOT_DIR "/opt/homebrew/anaconda3/bin/python")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/..)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB my_src CONFIGURE_DEPENDS
	"siclib/*.cpp"
	"siclib/*.hpp"
)

file(GLOB nest_src CONFIGURE_DEPENDS
	"siclib/**/*.cpp"
	"siclib/**/*.hpp"
)


pybind11_add_module(
	_pysiclib ${my_src} ${nest_src})

if(APPLE)
	target_link_libraries(_pysiclib PRIVATE ${CPU_BLAS})
else()
	target_link_libraries(_pysiclib PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# add_executable(main.out tests/main.cpp ${nest_src})

# add_test(test1 main.out)
