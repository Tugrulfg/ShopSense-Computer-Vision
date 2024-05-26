cmake_minimum_required(VERSION 3.18 FATAL_ERROR)  # Use a higher minimum version for better compatibility

project(Track)

# Find OpenCV (adjust the path if needed)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found. Please install OpenCV or adjust the PATH.")
endif()

find_package(OpenMP REQUIRED)

# Include necessary header directories
include_directories(${OpenCV_INCLUDE_DIRS} )

set(SOURCES track.cpp Inference.cpp)  

# Create the executable
add_executable(Track ${SOURCES})

# Link against PyTorch and OpenCV libraries
target_link_libraries(Track OpenMP::OpenMP_CXX ${OpenCV_LIBS})