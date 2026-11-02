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

#pragma once

#include <cstdint>
#include "types.h"

namespace turbo_ops_lora {
  // Combined LoRA gather index for the AllGather MoE path: aten builds the
  // unique fp32 sort keys + at::argsort, this kernel folds the entire
  // post-sort chain (gathers/pad/clamp/index/where) into one pass
  extern void combined_lora_idx_finalize_impl(
        void *stream,
        void *dest,
        void *topk,
        void *lora_indices,
        void *adapter_enabled,
        void *inv,
        void *out,
        uint32_t num_pairs,
        uint32_t top_k,
        uint32_t num_valid_tokens,
        int64_t first_expert_idx,
        uint32_t num_experts,
        uint32_t aiv_num,
        bool dest_is_int32,
        bool topk_is_int32);

  extern void add_lora_fused_impl(
        AscendType type,
        void *stream,
        void *x,
        void *const *wa,
        void *const *wb,
        void *indices,
        void *y,
        void *z1ws,
        uint32_t batch_size,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        const uint32_t *h2,
        uint32_t n_slices,
        uint32_t y_width,
        float scale,
        uint32_t add_inputs,
        uint32_t aiv_num);

  extern void add_lora_swiglu_quant_impl(
        AscendType type,
        void *stream,
        void *gate_up,
        void *delta,
        void *act,
        void *y,
        void *scale,
        uint32_t batch,
        uint32_t width,
        uint32_t has_delta,
        float swiglu_limit,
        uint32_t aiv_num);

  extern void bgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y, 
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void bgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);

    extern void sgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void sgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);

}  // namespace turbo_ops_lora
