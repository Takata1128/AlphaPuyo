cmake_minimum_required(VERSION 2.8)
project(NumberDetector)

# Create Main project
add_executable(NumberDetector
    main.cpp
    tf_utils.cpp
    tf_utils.hpp
)

# For Tensorflow
if(WIN32)
    target_link_libraries(NumberDetector ${PROJECT_SOURCE_DIR}/libtensorflow/lib/tensorflow.lib)
    target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/libtensorflow/include)
    target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/3rd_party/include)
    file(COPY ${PROJECT_SOURCE_DIR}/libtensorflow/lib/tensorflow.dll DESTINATION ${PROJECT_BINARY_DIR}/Debug)
else()
    target_link_libraries(NumberDetector ${PROJECT_SOURCE_DIR}/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0/lib/libtensorflow.so)
    target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0/include)
    file(COPY ${PROJECT_SOURCE_DIR}/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0/lib/libtensorflow.so DESTINATION ${PROJECT_BINARY_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++")
endif()

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resources/ DESTINATION ${PROJECT_BINARY_DIR}/resources/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resources/")