cmake_minimum_required(VERSION 2.8)
project(jetson_cpp_opencv)

# https://stackoverflow.com/questions/11588855/how-do-you-set-cmake-c-compiler-and-cmake-cxx-compiler-for-building-assimp-for-i
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 11)

MESSAGE(WARNING "install path ${OpenCV_INSTALL_PATH}")
MESSAGE(WARNING "config path ${OpenCV_CONFIG_PATH}") # if this is in cache, find_package() below won't overwrite it

# http://docs.opencv.org/trunk/db/df5/tutorial_linux_gcc_cmake.html
find_package( OpenCV REQUIRED )
#find_package(OpenCV 2.4 REQUIRED) # change this to 3 after running cmake once
MESSAGE(WARNING "version ${OpenCV_VERSION}")
MESSAGE(WARNING "install path ${OpenCV_INSTALL_PATH}")
MESSAGE(WARNING "config path ${OpenCV_CONFIG_PATH}") # look at the output of this message
MESSAGE(WARNING "include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(WARNING "libs ${OpenCV_LIBS}")



include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES
        main.cpp
        )


add_executable(jetson_cpp_opencv ${SOURCE_FILES})
target_link_libraries( jetson_cpp_opencv ${OpenCV_LIBS} )