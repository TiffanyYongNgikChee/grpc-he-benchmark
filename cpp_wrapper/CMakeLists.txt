cmake_minimum_required(VERSION 3.13)
project(seal_wrapper)

# Find SEAL
find_package(SEAL 4.1 REQUIRED)

# Create shared library
add_library(seal_wrapper SHARED
    src/seal_wrapper.cpp
)

# Link SEAL
target_link_libraries(seal_wrapper SEAL::seal)

# Include directories
target_include_directories(seal_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set C++17 standard
set_property(TARGET seal_wrapper PROPERTY CXX_STANDARD 17)

# Install library
install(TARGETS seal_wrapper
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/seal_wrapper.h
    DESTINATION include
)