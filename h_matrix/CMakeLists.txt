# for gtest_add_tests
cmake_minimum_required(VERSION 3.3)

project(nbody_benchmark_h_matrix VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-padded -Wno-double-promotion -stdlib=libc++ -fopenmp")
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

# main target
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(nbody_benchmark_h_matrix src/main.cpp)
target_link_libraries(nbody_benchmark_h_matrix lapacke)

# test target
if(BUILD_TEST)
    include_directories(${EP_PREFIX}/install/googletest/include)
    link_directories(${EP_PREFIX}/install/googletest/lib)

    add_executable(test_problem_h_matrix test/test_problem.cpp)
    add_dependencies(test_problem_h_matrix googletest)
    target_link_libraries(test_problem_h_matrix gtest gtest_main pthread lapacke)
    gtest_add_tests(TARGET test_problem_h_matrix)

    add_executable(test_matrix_h_matrix test/test_matrix.cpp)
    add_dependencies(test_matrix_h_matrix googletest)
    target_link_libraries(test_matrix_h_matrix gtest gtest_main pthread lapacke)
    gtest_add_tests(TARGET test_matrix_h_matrix)

    add_executable(test_solver_ref_h_matrix test/test_solver_ref.cpp)
    add_dependencies(test_solver_ref_h_matrix googletest)
    target_link_libraries(test_solver_ref_h_matrix gtest gtest_main pthread lapacke)
    gtest_add_tests(TARGET test_solver_ref_h_matrix)

    add_executable(test_benchmark_h_matrix test/test_benchmark.cpp)
    add_dependencies(test_benchmark_h_matrix googletest)
    target_link_libraries(test_benchmark_h_matrix gtest gtest_main pthread lapacke)
    gtest_add_tests(TARGET test_benchmark_h_matrix)
endif()
