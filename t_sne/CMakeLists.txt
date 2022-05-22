# for gtest_add_tests
cmake_minimum_required(VERSION 3.3)

project(nbody_benchmark_t_sne VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-padded -stdlib=libc++ -fopenmp")
set(CMAKE_CXX_GOOGLETEST_FLAGS "-stdlib=libc++")

option(BUILD_TEST "build test" OFF)
if(BUILD_TEST)
    # test settings
    enable_testing()

    include(ExternalProject)
    set(EP_PREFIX ${CMAKE_BINARY_DIR}/external)

    externalproject_add(
        googletest
        SOURCE_DIR ${EP_PREFIX}/src/googletest
        BINARY_DIR ${EP_PREFIX}/build/googletest  
        INSTALL_DIR ${EP_PREFIX}/install/googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG release-1.10.0
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EP_PREFIX}/install/googletest -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_GOOGLETEST_FLAGS}
    )
    include(GoogleTest)
endif()
option(BUILD_DOCUMENT "generate document by doxygen" OFF)
if(BUILD_DOCUMENT)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        message(${CMAKE_CURRENT_SOURCE_DIR})
        set(DOXYGEN_CONF_FILE "doxygen.conf")
        set(DOXYGEN_CONF_PATH ${CMAKE_CURRENT_BINARY_DIR}/${DOXYGEN_CONF_FILE})
        set(DOXYGEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        set(DOXYGEN_TARGET "docs")
        configure_file(${DOXYGEN_CONF_FILE}.in ${DOXYGEN_CONF_PATH})
        add_custom_target(${DOXYGEN_TARGET} ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONF_PATH} DEPENDS ${DOXYGEN_CONF_PATH})
    else()
        message(WARNING "doxygen not found")
    endif()
endif()

# data
function(download_gz URL GZ_FILE)
    set(GZ_FILE ${CMAKE_CURRENT_BINARY_DIR}/${GZ_FILE})
    string(REGEX REPLACE "\\.[^.]*$" "" FILE ${GZ_FILE})
    if(NOT EXISTS ${FILE})
        if(NOT EXISTS ${GZ_FILE})
            message("download ${URL} to ${GZ_FILE}")
            file(DOWNLOAD ${URL} ${GZ_FILE} SHOW_PROGRESS STATUS DOWNLOAD_STATUS)
            list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
            list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
            if(NOT ${STATUS_CODE} EQUAL 0)
                message(FATAL_ERROR "Error occurred during download ${URL}: ${ERROR_MESSAGE}")
            endif()
        endif()
        message("unpack ${GZ_FILE} to ${FILE}")
        execute_process(COMMAND gunzip ${GZ_FILE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} RESULT_VARIABLE UNZIP_STATUS)
        if(NOT ${UNZIP_STATUS} EQUAL 0)
            message(FATAL_ERROR "Error occurred during unpack ${GZ_FILE}")
        endif()
    endif()
endfunction()

set(MNIST_DATASET_URL_BASE "https://web.archive.org/web/20210304042348/http://yann.lecun.com/exdb/mnist/")
download_gz("${MNIST_DATASET_URL_BASE}train-images-idx3-ubyte.gz" data/train-images-idx3-ubyte.gz)
download_gz("${MNIST_DATASET_URL_BASE}train-labels-idx1-ubyte.gz" data/train-labels-idx1-ubyte.gz)
download_gz("${MNIST_DATASET_URL_BASE}t10k-images-idx3-ubyte.gz" data/t10k-images-idx3-ubyte.gz)
download_gz("${MNIST_DATASET_URL_BASE}t10k-labels-idx1-ubyte.gz" data/t10k-labels-idx1-ubyte.gz)


# main target
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(nbody_benchmark_t_sne src/main.cpp)

# test target
if(BUILD_TEST)
    include_directories(${EP_PREFIX}/install/googletest/include)
    link_directories(${EP_PREFIX}/install/googletest/lib)

    add_executable(test_problem_t_sne test/test_problem.cpp)
    add_dependencies(test_problem_t_sne googletest)
    target_link_libraries(test_problem_t_sne gtest gtest_main pthread)
    gtest_add_tests(TARGET test_problem_t_sne)

    add_executable(test_solver_ref_naive_t_sne test/test_solver_ref_naive.cpp)
    add_dependencies(test_solver_ref_naive_t_sne googletest)
    target_link_libraries(test_solver_ref_naive_t_sne gtest gtest_main pthread)
    gtest_add_tests(TARGET test_solver_ref_naive_t_sne)

    add_executable(test_benchmark_t_sne test/test_benchmark.cpp)
    add_dependencies(test_benchmark_t_sne googletest)
    target_link_libraries(test_benchmark_t_sne gtest gtest_main pthread)
    gtest_add_tests(TARGET test_benchmark_t_sne)

    add_executable(test_barnes_hut_tree_t_sne test/test_barnes_hut_tree.cpp)
    add_dependencies(test_barnes_hut_tree_t_sne googletest)
    target_link_libraries(test_barnes_hut_tree_t_sne gtest gtest_main pthread)
    gtest_add_tests(TARGET test_barnes_hut_tree_t_sne)

    add_executable(test_vantage_point_tree_t_sne test/test_vantage_point_tree.cpp)
    add_dependencies(test_vantage_point_tree_t_sne googletest)
    target_link_libraries(test_vantage_point_tree_t_sne gtest gtest_main pthread)
    gtest_add_tests(TARGET test_vantage_point_tree_t_sne)

    add_executable(test_solver_ref_t_sne test/test_solver_ref.cpp)
    add_dependencies(test_solver_ref_t_sne googletest)
    target_link_libraries(test_solver_ref_t_sne gtest gtest_main pthread)
    gtest_add_tests(TARGET test_solver_ref_t_sne)
endif()
