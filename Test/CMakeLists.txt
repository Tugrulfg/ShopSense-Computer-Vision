cmake_minimum_required(VERSION 3.18 FATAL_ERROR)  # Use a higher minimum version for better compatibility

project(Track)

# # Find PyTorch (adjust the path if needed)
find_package(Torch REQUIRED PATHS /home/tugrul/libtorch)
if(NOT Torch_FOUND)
  message(FATAL_ERROR "PyTorch not found. Please install PyTorch or adjust the PATH.")
endif()


# Include necessary header directories
include_directories(${TORCH_INCLUDE_DIRS} )


set(SOURCES test.cpp)  

# Create the executable
add_executable(Test ${SOURCES})

target_link_libraries(Test ${TORCH_LIBRARIES})
