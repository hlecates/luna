# CMakeLists.txt for lirpapy Python bindings

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create Python module
pybind11_add_module(LirpaPyCore LirpaPyCore.cpp)

# Link against LIRPACore and dependencies
target_link_libraries(LirpaPyCore PRIVATE ${LIRPA_LIB} ${LIBS})

# Include directories
target_include_directories(LirpaPyCore PRIVATE
    ${SRC_DIR}
    ${COMMON_DIR}
    ${ENGINE_DIR}
    ${INPUT_PARSERS_DIR}
    ${LIBS_INCLUDES}
)

# Set output directory for the Python module
set_target_properties(LirpaPyCore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Install the module
install(TARGETS LirpaPyCore
    LIBRARY DESTINATION ${Python3_SITELIB}/lirpapy
    COMPONENT python
)

message(STATUS "Configured lirpapy Python bindings")
