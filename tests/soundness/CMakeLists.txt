# tests/soundness/CMakeLists.txt
# Soundness test configuration

set(SOUNDNESS_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_soundness_test TEST_NAME)
    lirpa_add_test(${TEST_NAME} ${TEST_NAME}.cpp "soundness" "system")
endfunction()

add_soundness_test(test_crown_soundness)
add_soundness_test(test_ibp_soundness)
add_soundness_test(test_relaxation_soundness)
add_soundness_test(test_alpha_crown_soundness)
