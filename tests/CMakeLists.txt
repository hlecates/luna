# tests/CMakeLists.txt
# Top-level test configuration for LIRPA testing framework

set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TESTS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TEST_DIR "${CMAKE_BINARY_DIR}/tests")

# Common include directories for all tests
set(TEST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/engine
    ${PROJECT_SOURCE_DIR}/src/engine/nodes
    ${PROJECT_SOURCE_DIR}/src/engine/conv
    ${PROJECT_SOURCE_DIR}/src/input_parsers
    ${PROJECT_SOURCE_DIR}/src/configuration
    ${LIBS_INCLUDES}
)

# Common compile flags for tests (slightly different from release)
if (NOT MSVC)
    set(TEST_COMPILE_FLAGS ${COMPILE_FLAGS})
    # Tests don't need -O3, use -O2 for faster compile times
    string(REPLACE "-O3" "-O2" TEST_COMPILE_FLAGS "${TEST_COMPILE_FLAGS}")
else()
    set(TEST_COMPILE_FLAGS "")
endif()

# Build CxxTest runner library
add_library(CxxTestRunner STATIC
    ${CXXTEST_DIR}/src/cxxtest_runner.cpp
    ${CXXTEST_DIR}/src/cxxtest_main.cpp
)
target_include_directories(CxxTestRunner PUBLIC ${CXXTEST_INCLUDE_DIR})

# Unified macro for registering tests
function(lirpa_add_test TEST_NAME TEST_SOURCE TEST_LABEL TEST_TIER)
    set(TEST_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SOURCE})
    get_filename_component(TEST_EXT ${TEST_SOURCE_PATH} EXT)

    if(TEST_EXT STREQUAL ".h")
        set(GENERATED_CC ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}_gen.cc)
        add_custom_command(
            OUTPUT ${GENERATED_CC}
            COMMAND ${CXXTEST_PYTHON_TESTGEN_EXECUTABLE} -o ${GENERATED_CC} ${TEST_SOURCE_PATH}
            DEPENDS ${TEST_SOURCE_PATH}
            COMMENT "Generating CxxTest runner for ${TEST_SOURCE}"
        )
        set(TEST_SOURCES ${GENERATED_CC})
    else()
        set(TEST_SOURCES ${TEST_SOURCE_PATH})
    endif()

    add_executable(${TEST_NAME} ${TEST_SOURCES})

    target_include_directories(${TEST_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}  # Project root for "src/..." includes
        ${PROJECT_SOURCE_DIR}/tests  # For "fixtures/..." includes
        ${TEST_INCLUDE_DIRS}
        ${CXXTEST_INCLUDE_DIR}
    )

    target_link_libraries(${TEST_NAME}
        TestFixtures
        ${LIRPA_LIB}
        CxxTestRunner
        ${LIBS}
    )

    target_compile_options(${TEST_NAME} PRIVATE ${TEST_COMPILE_FLAGS})

    add_test(NAME ${TEST_NAME}
        COMMAND ${TEST_NAME}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "${TEST_LABEL};${TEST_TIER}")

    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_DIR}/${TEST_TIER}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TEST_NAME}> ${TEST_DIR}/${TEST_TIER}/
    )

    set_property(GLOBAL APPEND PROPERTY LIRPA_TEST_TARGETS ${TEST_NAME})
endfunction()

if(RUN_UNIT_TEST OR RUN_SYSTEM_TEST OR RUN_REGRESS_TEST)
    add_subdirectory(fixtures)
endif()

if(RUN_UNIT_TEST)
    add_subdirectory(unit)
    add_subdirectory(property)
endif()

if(RUN_SYSTEM_TEST)
    add_subdirectory(integration)
    add_subdirectory(soundness)
endif()

if(RUN_REGRESS_TEST)
    add_subdirectory(regression)
    add_subdirectory(reference)
endif()

# Build benchmarks if requested
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(ProcessorCount)
ProcessorCount(CTEST_NTHREADS)
if(NOT CTEST_NTHREADS OR CTEST_NTHREADS EQUAL 0)
    set(CTEST_NTHREADS 1)
endif()

get_property(LIRPA_TEST_TARGETS GLOBAL PROPERTY LIRPA_TEST_TARGETS)
if(LIRPA_TEST_TARGETS)
    add_custom_target(build-tests DEPENDS ${LIRPA_TEST_TARGETS})
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CTEST_NTHREADS} $$ARGS
        DEPENDS build-tests
    )
    add_custom_target(systemtests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CTEST_NTHREADS} -L system $$ARGS
        DEPENDS build-tests
    )
    add_custom_target(regress
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CTEST_NTHREADS} -L regress $$ARGS
        DEPENDS build-tests
    )
endif()

if(RUN_PYTHON_TEST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(pytests
        COMMAND ${Python3_EXECUTABLE} ${TESTS_DIR}/regress/run_regression.py $$ARGS
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif()
