/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

namespace turbo_ops_lora {
namespace meta {

void bgmv_shrink_meta(at::Tensor &x, at::Tensor &weight, at::Tensor &indices,
                      at::Tensor &y, double scale) {}

void sgmv_shrink_meta(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices,
                      at::Tensor &seq_len, at::Tensor &y, double scale) {}

at::Tensor bgmv_expand_meta(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                        int64_t slice_offset, int64_t slice_size) {
    return y;
}

at::Tensor sgmv_expand_meta(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                        at::Tensor &y, int64_t slice_offset, int64_t slice_size) {
    return y;
}

// Fused LoRA apply ops are mutate-only (-> ()): nothing to fake-allocate.
void add_lora_shrink_meta(std::vector<at::Tensor> y, at::Tensor x, std::vector<at::Tensor> lora_a,
                          at::Tensor lora_indices, at::Tensor seq_len, at::Tensor token_lora_indices,
                          double scale, at::Tensor use_gmm, at::Tensor no_lora,
                          bool is_moe = false, int64_t group_list_type = 1) {}

void add_lora_expand_meta(at::Tensor y, std::vector<at::Tensor> x, std::vector<at::Tensor> lora_b,
                          at::Tensor lora_indices, at::Tensor seq_len, at::Tensor token_lora_indices,
                          std::vector<int64_t> output_slices, int64_t offset_start, bool add_inputs,
                          at::Tensor use_gmm, at::Tensor no_lora,
                          bool is_moe = false, int64_t group_list_type = 1) {}

void add_lora_meta(at::Tensor y, at::Tensor x, std::vector<at::Tensor> lora_a,
                   std::vector<at::Tensor> lora_b, at::Tensor lora_indices, at::Tensor seq_len,
                   at::Tensor token_lora_indices, std::vector<int64_t> output_slices,
                   int64_t offset_start, double scale, bool add_inputs,
                   at::Tensor use_gmm, at::Tensor no_lora, at::Tensor use_add_lora) {}

at::Tensor add_lora_expand_delta_meta(const at::Tensor &gate_up, std::vector<at::Tensor> x,
                                      std::vector<at::Tensor> lora_b, at::Tensor seq_len,
                                      int64_t group_list_type = 1)
{
    return at::empty_symint({gate_up.sym_size(0), gate_up.sym_size(1)}, gate_up.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> add_lora_swiglu_quant_meta(
    const at::Tensor &gate_up, const c10::optional<at::Tensor> &delta, double swiglu_limit = 0.0)
{
    // sym-safe: dynamo traces a dynamic token count through here
    auto batch = gate_up.sym_size(0);
    auto width = gate_up.sym_size(1) / 2;
    return {at::empty_symint({batch, width}, gate_up.options()),
            at::empty_symint({batch, width}, gate_up.options().dtype(at::kChar)),
            at::empty_symint({batch}, gate_up.options().dtype(at::kFloat))};
}

at::Tensor build_combined_lora_idx_meta(at::Tensor dest, at::Tensor topk_ids,
                                        at::Tensor lora_indices, at::Tensor adapter_enabled,
                                        int64_t first_expert_idx, int64_t num_experts,
                                        int64_t top_k)
{
    // sym-safe numel: dynamo traces dynamic shapes through here
    return at::empty_symint({dest.sym_numel()}, dest.options().dtype(at::kLong));
}

}  // namespace meta
}  // namespace turbo_ops_lora

TORCH_LIBRARY_IMPL(turbo_ops, Meta, ops) {
    ops.impl("bgmv_shrink", &turbo_ops_lora::meta::bgmv_shrink_meta);
    ops.impl("bgmv_expand", &turbo_ops_lora::meta::bgmv_expand_meta);
    ops.impl("sgmv_shrink", &turbo_ops_lora::meta::sgmv_shrink_meta);
    ops.impl("sgmv_expand", &turbo_ops_lora::meta::sgmv_expand_meta);
    ops.impl("add_lora_shrink", &turbo_ops_lora::meta::add_lora_shrink_meta);
    ops.impl("add_lora_expand", &turbo_ops_lora::meta::add_lora_expand_meta);
    ops.impl("add_lora", &turbo_ops_lora::meta::add_lora_meta);
    ops.impl("add_lora_swiglu_quant", &turbo_ops_lora::meta::add_lora_swiglu_quant_meta);
    ops.impl("add_lora_expand_delta", &turbo_ops_lora::meta::add_lora_expand_delta_meta);
    ops.impl("build_combined_lora_idx", &turbo_ops_lora::meta::build_combined_lora_idx_meta);
}
