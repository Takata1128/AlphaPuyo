cmake_minimum_required(VERSION 2.8)
project(NumberDetector)

# Create Main project
add_executable(NumberDetector
    define.hpp
    puyoSim.cpp
    tf_utils.cpp
    tf_utils.hpp
    mcts.cpp
    mcts.hpp
    PuyoGame.cpp
    PuyoGame.hpp
)

# For Tensorflow
target_link_libraries(NumberDetector ${PROJECT_SOURCE_DIR}/libtensorflow/lib/tensorflow.lib)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/libtensorflow/include)
target_include_directories(NumberDetector PUBLIC ${PROJECT_SOURCE_DIR}/3rd_party/include)
file(COPY ${PROJECT_SOURCE_DIR}/libtensorflow/lib/tensorflow.dll DESTINATION ${PROJECT_BINARY_DIR}/Debug)

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resources/ DESTINATION ${PROJECT_BINARY_DIR}/resources/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resources/")