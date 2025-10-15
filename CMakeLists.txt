cmake_minimum_required(VERSION 3.31.0)
project(libtorch_examples)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(libtorch_VERSION 2.8.0)
find_package(Torch ${libtorch_VERSION} PATHS ./libtorch)
if(NOT Torch_FOUND)
  
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()

macro(add_example name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ./libtorch/include)
  target_include_directories(${name} PRIVATE ../libtorch/include/torch/csrc/api/include)
  target_link_libraries(${name} ${TORCH_LIBRARIES})
  target_compile_features(${name} PRIVATE cxx_std_17)
endmacro()

add_example(eye)