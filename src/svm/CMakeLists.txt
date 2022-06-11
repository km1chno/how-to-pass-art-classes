include_directories(../../lib/)
add_library(svm_lib svm.h svm.cpp)

target_include_directories(svm_lib
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )


