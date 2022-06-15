add_library(ada_lib adaboost.h adaboost.cpp)

target_include_directories(ada_lib
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )


