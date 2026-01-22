# src/common/CMakeLists.txt
# Common module sources

# Add common source files
file(GLOB SRCS_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
target_sources(${LIRPA_LIB} PRIVATE ${SRCS_COMMON})

# Add real subdirectory sources
file(GLOB SRCS_COMMON_REAL "${CMAKE_CURRENT_SOURCE_DIR}/real/*.cpp")
target_sources(${LIRPA_LIB} PRIVATE ${SRCS_COMMON_REAL})

# Set include directories for common module
target_include_directories(${LIRPA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

