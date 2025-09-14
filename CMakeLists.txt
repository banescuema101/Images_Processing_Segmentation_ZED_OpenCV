cmake_minimum_required(VERSION 3.10)
project(ZED_SVO_Playback)

set(CMAKE_CXX_STANDARD 17)

add_executable(ZED_SVO_Playback
    main.cpp
    Filter.cpp
    Filter.h
    utils.hpp
)
