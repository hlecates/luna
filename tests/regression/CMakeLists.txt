# tests/regression/CMakeLists.txt
# Regression test configuration

set(REGRESSION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_regression_test TEST_NAME)
    lirpa_add_test(${TEST_NAME} ${TEST_NAME}.cpp "regression" "regress")
endfunction()

add_regression_test(test_acasxu_regression)
add_regression_test(test_cifar_regression)
add_regression_test(test_mnist_regression)
add_regression_test(test_toy_models_regression)
