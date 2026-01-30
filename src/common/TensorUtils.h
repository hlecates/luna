#ifndef __TensorUtils_h__
#define __TensorUtils_h__

#include "configuration/LunaConfiguration.h"
#include <torch/torch.h>

namespace LirpaTensorUtils {

inline torch::TensorOptions defaultOptions()
{
    return torch::TensorOptions().dtype(torch::kFloat32).device(LunaConfiguration::getDevice());
}

inline torch::Tensor toDevice(const torch::Tensor &tensor)
{
    return tensor.to(LunaConfiguration::getDevice());
}

inline torch::Tensor zerosLikeOnDevice(const torch::Tensor &reference)
{
    return torch::zeros_like(reference, reference.options().device(LunaConfiguration::getDevice()));
}

inline torch::Tensor onesLikeOnDevice(const torch::Tensor &reference)
{
    return torch::ones_like(reference, reference.options().device(LunaConfiguration::getDevice()));
}

inline torch::Tensor emptyLikeOnDevice(const torch::Tensor &reference)
{
    return torch::empty_like(reference, reference.options().device(LunaConfiguration::getDevice()));
}

} // namespace LirpaTensorUtils

#endif // __TensorUtils_h__
