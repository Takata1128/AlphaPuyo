cmake_minimum_required(VERSION 2.8)
project(AlphaPuyo)

# Create Main project
add_executable(AlphaPuyo
    define.hpp
    puyoSim.cpp
    tf_utils.cpp
    tf_utils.hpp
    mcts.cpp
    mcts.hpp
    PuyoGame.cpp
    PuyoGame.hpp
)

set(CMAKE_CXX_FLAGS "/std:c++17")
set(BOOST_ROOT "D:/Boost")
set(PYTHON_ROOT "C:/Users/rokahikou/AppData/Local/Programs/Python/Python36")

# For Tensorflow
target_link_libraries(AlphaPuyo ${PROJECT_SOURCE_DIR}/libtensorflow/lib/tensorflow.lib)
target_include_directories(AlphaPuyo PUBLIC ${PROJECT_SOURCE_DIR}/libtensorflow/include)
target_include_directories(AlphaPuyo PUBLIC ${PROJECT_SOURCE_DIR}/3rd_party/include)
file(COPY ${PROJECT_SOURCE_DIR}/libtensorflow/lib/tensorflow.dll DESTINATION ${PROJECT_BINARY_DIR}/Debug)

#for python
target_link_libraries(AlphaPuyo ${PYTHON_ROOT}/libs/python36.lib)
target_include_directories(AlphaPuyo PUBLIC ${PYTHON_ROOT}/include)

# For Boost
target_link_libraries(AlphaPuyo ${BOOST_ROOT}/lib/libboost_python36-vc142-mt-x64-1_74.lib)
target_link_libraries(AlphaPuyo ${BOOST_ROOT}/lib/libboost_numpy36-vc142-mt-x64-1_74.lib)
target_include_directories(AlphaPuyo PUBLIC ${BOOST_ROOT}/include/boost-1_74)

# Copy resource
file(COPY ${CMAKE_SOURCE_DIR}/resources/ DESTINATION ${PROJECT_BINARY_DIR}/resources/)
file(COPY ${CMAKE_SOURCE_DIR}/save.py DESTINATION ${PROJECT_BINARY_DIR}/Release/)
file(COPY ${CMAKE_SOURCE_DIR}/libtensorflow/lib/tensorflow.dll DESTINATION ${PROJECT_BINARY_DIR}/Release/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resources/")