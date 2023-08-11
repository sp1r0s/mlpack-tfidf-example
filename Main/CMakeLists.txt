project(Train LANGUAGES CXX)

add_executable(Train
               src/Train.cpp
               )

find_package(Armadillo REQUIRED)
find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")
target_include_directories(Train PRIVATE ${MLPACK_INCLUDE_DIRS})
target_link_libraries(Train ${ARMADILLO_LIBRARIES})
