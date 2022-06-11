cmake_minimum_required(VERSION 3.22)
project(how_to_pass_art_classes)

set(CMAKE_CXX_STANDARD 20)

add_definitions(-pthread -O3)
set(GCC_COVERAGE_LINK_FLAGS "-lm -lpthread -lX11 -ljpeg ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

include_directories(../lib)

add_subdirectory(images)
add_subdirectory(pca)
add_subdirectory(svm)
add_subdirectory(hog)

set(SOURCE_FILES main_pca_hog.cpp tools.h tools.cpp)
set(SOURCE_FILES2 main_generate_paintdata.cpp)
set(SOURCE_FILES3 main_pca_svm.cpp tools.h tools.cpp)
set(SCRYPT_NAME out_images_scrypt)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_executable(${SCRYPT_NAME} ${SOURCE_FILES2})
add_executable(pca_svm_test ${SOURCE_FILES3})

target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}/lib
        ${CMAKE_CURRENT_LIST_DIR}
        )

target_link_libraries(${PROJECT_NAME} images_helper)
target_link_libraries(${PROJECT_NAME} pca_lib)
target_link_libraries(${PROJECT_NAME} svm_lib)
target_link_libraries(${PROJECT_NAME} hog_lib)


target_link_libraries(${SCRYPT_NAME} images_helper)
target_link_libraries(${SCRYPT_NAME} pca_lib)
target_link_libraries(${SCRYPT_NAME} svm_lib)

target_link_libraries(pca_svm_test images_helper)
target_link_libraries(pca_svm_test  pca_lib)
target_link_libraries(pca_svm_test svm_lib)
target_link_libraries(pca_svm_test hog_lib)


