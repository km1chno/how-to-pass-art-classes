cmake_minimum_required(VERSION 3.22)
project(how_to_pass_art_classes)

set(CMAKE_CXX_STANDARD 20)

set(GCC_COVERAGE_LINK_FLAGS "-lm -lpthread -lX11 -ljpeg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

include_directories(../lib)

add_subdirectory(images)
add_subdirectory(pca)
add_subdirectory(svm)

set(SOURCE_FILES main.cpp)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}/lib
        ${CMAKE_CURRENT_LIST_DIR}
        )
target_link_libraries(${PROJECT_NAME} images_helper)
target_link_libraries(${PROJECT_NAME} pca_lib)
target_link_libraries(${PROJECT_NAME} svm_lib)



