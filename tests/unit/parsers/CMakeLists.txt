# tests/unit/parsers/CMakeLists.txt
# Parser unit tests

function(add_parser_test TEST_NAME)
    lirpa_add_test(${TEST_NAME} ${TEST_NAME}.cpp "parsers" "unit")
endfunction()

add_parser_test(test_onnx_parser)
add_parser_test(test_vnnlib_parser)
