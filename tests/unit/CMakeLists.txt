# tests/unit/CMakeLists.txt
# Unit test configuration

set(UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Build node tests
add_subdirectory(nodes)

# Build analysis tests
add_subdirectory(analysis)

# Build parser tests
add_subdirectory(parsers)
