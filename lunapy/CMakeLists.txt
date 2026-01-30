# CMakeLists.txt for lunapy Python bindings

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create Python module
pybind11_add_module(LunaPyCore LunaPyCore.cpp)

# Link against LUNACore and dependencies
target_link_libraries(LunaPyCore PRIVATE ${LUNA_LIB} ${LIBS})

# Include directories
target_include_directories(LunaPyCore PRIVATE
    ${SRC_DIR}
    ${COMMON_DIR}
    ${ENGINE_DIR}
    ${INPUT_PARSERS_DIR}
    ${LIBS_INCLUDES}
)

# Set output directory for the Python module
set_target_properties(LunaPyCore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Install the module
install(TARGETS LunaPyCore
    LIBRARY DESTINATION ${Python3_SITELIB}/lunapy
    COMPONENT python
)

message(STATUS "Configured lunapy Python bindings")
