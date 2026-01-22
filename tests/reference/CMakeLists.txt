# tests/reference/CMakeLists.txt
# Reference test configuration (cross-validation with auto_LiRPA)

set(REFERENCE_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_reference_test TEST_NAME)
    lirpa_add_test(${TEST_NAME} ${TEST_NAME}.cpp "reference" "regress")
endfunction()

add_reference_test(test_crown_vs_autolirpa)
add_reference_test(test_alpha_crown_vs_autolirpa)
add_reference_test(test_comprehensive_reference)
