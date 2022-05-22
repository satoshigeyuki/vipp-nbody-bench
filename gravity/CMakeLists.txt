cmake_minimum_required(VERSION 3.3)

project(nbody_benchmark_gravity VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-padded -Wno-double-promotion -Werror -Wno-error=implicit-int-float-conversion -stdlib=libc++ -fopenmp")
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
add_executable(nbody_benchmark_gravity src/main.cpp)

# test target
if(BUILD_TEST)
    include_directories(${EP_PREFIX}/install/googletest/include)
    link_directories(${EP_PREFIX}/install/googletest/lib)

    add_executable(test_problem_gravity test/test_problem.cpp)
    add_dependencies(test_problem_gravity googletest)
    target_link_libraries(test_problem_gravity gtest gtest_main pthread)
    gtest_add_tests(TARGET test_problem_gravity)

    add_executable(test_solver_ref_naive_gravity test/test_solver_ref_naive.cpp)
    add_dependencies(test_solver_ref_naive_gravity googletest)
    target_link_libraries(test_solver_ref_naive_gravity gtest gtest_main pthread)
    gtest_add_tests(TARGET test_solver_ref_naive_gravity)

    add_executable(test_benchmark_gravity test/test_benchmark.cpp)
    add_dependencies(test_benchmark_gravity googletest)
    target_link_libraries(test_benchmark_gravity gtest gtest_main pthread)
    gtest_add_tests(TARGET test_benchmark_gravity)

    add_executable(test_octree_gravity test/test_octree.cpp)
    add_dependencies(test_octree_gravity googletest)
    target_link_libraries(test_octree_gravity gtest gtest_main pthread)
    gtest_add_tests(TARGET test_octree_gravity)

    add_executable(test_fmm_gravity test/test_fmm.cpp)
    add_dependencies(test_fmm_gravity googletest)
    target_link_libraries(test_fmm_gravity gtest gtest_main pthread)
    gtest_add_tests(TARGET test_fmm_gravity)

    add_executable(test_solver_ref_gravity test/test_solver_ref.cpp)
    add_dependencies(test_solver_ref_gravity googletest)
    target_link_libraries(test_solver_ref_gravity gtest gtest_main pthread)
    gtest_add_tests(TARGET test_solver_ref_gravity)
endif()
