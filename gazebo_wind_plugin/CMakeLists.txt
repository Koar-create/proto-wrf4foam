cmake_minimum_required(VERSION 3.10)
project(gazebo_wind_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gazebo REQUIRED)

# VTK 9.x on Ubuntu 22.04 typically provides targets like VTK::IOXML, etc.
find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  IOXML
)

add_library(WindFieldPlugin SHARED
  WindFieldPlugin.cc
  lut_reader/WindLUT.cc
)

target_include_directories(WindFieldPlugin PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(WindFieldPlugin
  ${GAZEBO_LIBRARIES}
  VTK::CommonCore
  VTK::CommonDataModel
  VTK::IOXML
)

target_compile_definitions(WindFieldPlugin PRIVATE
  GAZEBO_WIND_PLUGIN_USE_VTK_VTI=1
)

add_library(HoverPidPlugin SHARED
  HoverPidPlugin.cc
)

target_include_directories(HoverPidPlugin PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(HoverPidPlugin
  ${GAZEBO_LIBRARIES}
)

add_library(TrailMarkerPlugin SHARED
  TrailMarkerPlugin.cc
)

target_include_directories(TrailMarkerPlugin PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(TrailMarkerPlugin
  ${GAZEBO_LIBRARIES}
)

add_library(ContactWatcherPlugin SHARED
  ContactWatcherPlugin.cc
)

target_include_directories(ContactWatcherPlugin PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ContactWatcherPlugin
  ${GAZEBO_LIBRARIES}
)

