cmake_minimum_required(VERSION 2.8)
project( transform_pointcloud )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( transform_pointcloud transform_pointcloud.cpp )
target_link_libraries( transform_pointcloud ${OpenCV_LIBS} )