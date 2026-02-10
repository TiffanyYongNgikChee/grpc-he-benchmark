cmake_minimum_required(VERSION 3.13)
project(openfhe_wrapper)

# Use C++17 standard (required by OpenFHE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE package
set(OpenFHE_DIR /usr/local/lib/OpenFHE)
find_package(OpenFHE REQUIRED)

# Print OpenFHE information for debugging
if(OpenFHE_FOUND)
    message(STATUS "   OpenFHE found")
    message(STATUS "   Version: ${OpenFHE_VERSION}")
    message(STATUS "   Include dirs: ${OpenFHE_INCLUDE}")
    message(STATUS "   Lib dirs: ${OpenFHE_LIBDIR}")
else()
    message(FATAL_ERROR " OpenFHE not found!")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# Link directories
link_directories(${OpenFHE_LIBDIR})

# Create shared library
add_library(openfhe_wrapper SHARED
    src/openfhe_wrapper.cpp
    src/openfhe_cnn_ops.cpp
)

# Link against OpenFHE libraries
target_link_libraries(openfhe_wrapper 
    OPENFHEpke
    OPENFHEcore
    OPENFHEbinfhe
)

# Set library properties
set_target_properties(openfhe_wrapper PROPERTIES
    VERSION 1.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "include/openfhe_wrapper.h;include/openfhe_cnn_ops.h"
)

# Installation rules
install(TARGETS openfhe_wrapper
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)