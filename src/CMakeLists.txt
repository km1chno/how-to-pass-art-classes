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
add_subdirectory(glcm)
add_subdirectory(trees)

set(PROJECT_SOURCE_FILES main.cpp)
set(SOURCE_FILES main_pca_hog.cpp)
set(SOURCE_FILES2 main_generate_paintdata.cpp)
set(SOURCE_FILES3 main_pca_svm.cpp tools.h tools.cpp)
set(SOURCE_FILES4 main_pca_glcm.cpp tools.h tools.cpp)
set(SOURCE_FILES5 main_trees.cpp tools.h tools.cpp)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})
add_executable(main_pca_hog ${SOURCE_FILES})
add_executable(main_generate_paintdata ${SOURCE_FILES2})
add_executable(main_pca_svm ${SOURCE_FILES3})
add_executable(main_pca_glcm ${SOURCE_FILES4})
add_executable(main_trees ${SOURCE_FILES5})

#target_include_directories(${PROJECT_NAME} PUBLIC
#        ${PROJECT_SOURCE_DIR}/lib
#        ${CMAKE_CURRENT_LIST_DIR}
#        )

target_link_libraries(main_pca_hog pca_lib)
target_link_libraries(main_pca_hog svm_lib)
target_link_libraries(main_pca_hog hog_lib)

target_link_libraries(main_generate_paintdata images_helper)
target_link_libraries(main_generate_paintdata pca_lib)
target_link_libraries(main_generate_paintdata svm_lib)

target_link_libraries(main_pca_svm pca_lib)
target_link_libraries(main_pca_svm svm_lib)
target_link_libraries(main_pca_svm hog_lib)

target_link_libraries(main_pca_glcm pca_lib)
target_link_libraries(main_pca_glcm svm_lib)
target_link_libraries(main_pca_glcm glcm_lib)

target_link_libraries(main_trees pca_lib)
target_link_libraries(main_trees svm_lib)
target_link_libraries(main_trees glcm_lib)
target_link_libraries(main_trees trees_lib)



