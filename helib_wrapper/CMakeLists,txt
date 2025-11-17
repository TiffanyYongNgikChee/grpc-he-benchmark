cmake_minimum_required(VERSION 3.13)
project(helib_wrapper)

# Find HElib
find_package(helib 2.2.0 REQUIRED)

# Find NTL
find_library(NTL_LIB ntl REQUIRED)
find_library(GMP_LIB gmp REQUIRED)

# Create shared library
add_library(helib_wrapper SHARED
    src/helib_wrapper.cpp
)

# Link HElib and dependencies
target_link_libraries(helib_wrapper 
    helib
    ${NTL_LIB}
    ${GMP_LIB}
    pthread
)

# Include directories
target_include_directories(helib_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set C++17 standard (HElib requires C++17)
set_property(TARGET helib_wrapper PROPERTY CXX_STANDARD 17)

# Install library
install(TARGETS helib_wrapper
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/helib_wrapper.h
    DESTINATION include
)