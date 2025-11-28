cmake_minimum_required(VERSION 3.13)
project(openfhe_wrapper)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenFHE
set(OpenFHE_DIR /usr/local/lib/OpenFHE)
find_package(OpenFHE CONFIG REQUIRED)

if(OpenFHE_FOUND)
    message(STATUS "Found OpenFHE")
    message(STATUS "OpenFHE include: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib: ${OpenFHE_LIBDIR}")
else()
    message(FATAL_ERROR "OpenFHE not found!")
endif()

# Include directories - add core subdirectory for relative includes
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link directories
link_directories(${OpenFHE_LIBDIR})

# Create shared library
add_library(openfhe_wrapper SHARED
    src/openfhe_wrapper.cpp
)

# Link OpenFHE libraries
target_link_libraries(openfhe_wrapper 
    OPENFHEpke 
    OPENFHEcore
    OPENFHEbinfhe
)

# Set library properties
set_target_properties(openfhe_wrapper PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    VERSION 1.0
    SOVERSION 1
)

# Install rules
install(TARGETS openfhe_wrapper
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/openfhe_wrapper.h
    DESTINATION include
)