# src/input_parsers/CMakeLists.txt
# Input parsers module sources

# Add input parser source files
file(GLOB SRCS_INPUT_PARSERS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
target_sources(${LUNA_LIB} PRIVATE ${SRCS_INPUT_PARSERS})

# Set include directories for input_parsers module
target_include_directories(${LUNA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

