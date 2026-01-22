#include "engine/LirpaMain.h"

int main(int argc, char** argv) {
    return lirpaMain(argc, argv);
}

// lirpa --input resources/onnx/spec_test.onnx --vnnlib resources/properties/spec_test.vnnlib

// lirpa --input resources/onnx/cifar_base_kw_simp.onnx --vnnlib resources/onnx/vnnlib/cifar_bounded.vnnlib

// lirpa --input resources/onnx/ACASXU_run2a_1_1_batch_2000.onnx --vnnlib resources/onnx/vnnlib/prop_1.vnnlib