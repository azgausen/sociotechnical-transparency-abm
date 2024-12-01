cmake_minimum_required(VERSION 3.22.1)
project(abm)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS -O2)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 11)


add_executable(abm main.cpp G.cpp)
