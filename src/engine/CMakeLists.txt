# src/engine/CMakeLists.txt
# Engine module sources

# Add engine source files (exclude LirpaMain.cpp as it's only for the executable)
file(GLOB SRCS_ENGINE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM SRCS_ENGINE "${CMAKE_CURRENT_SOURCE_DIR}/LirpaMain.cpp")
target_sources(${LIRPA_LIB} PRIVATE ${SRCS_ENGINE})

# Add conv subdirectory sources
file(GLOB SRCS_ENGINE_CONV "${CMAKE_CURRENT_SOURCE_DIR}/conv/*.cpp")
target_sources(${LIRPA_LIB} PRIVATE ${SRCS_ENGINE_CONV})

# Add nodes subdirectory sources
file(GLOB SRCS_ENGINE_NODES "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp")
target_sources(${LIRPA_LIB} PRIVATE ${SRCS_ENGINE_NODES})

# Set include directories for engine module and subdirectories
target_include_directories(${LIRPA_LIB} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/conv
    ${CMAKE_CURRENT_SOURCE_DIR}/nodes
)

