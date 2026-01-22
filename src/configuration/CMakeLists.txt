# src/configuration/CMakeLists.txt
# Configuration module sources

# Add configuration source files
target_sources(${LIRPA_LIB} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LirpaConfiguration.cpp
)

# Set include directories for configuration module
target_include_directories(${LIRPA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

