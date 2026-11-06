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
#include <torch/torch.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <array>
#include <algorithm>
#include <vector>
#include "ops.h"
#include "aclnn_torch_adapter/op_api_common.h"

namespace turbo_ops_lora {
using namespace aclnn;

AscendType get_dtype_from_torch(at::ScalarType scalarType)
{
    if (scalarType == at::ScalarType::Float) {
        return AscendType::FP32;
    } else if (scalarType == at::ScalarType::BFloat16) {
        return AscendType::BF16;
    } else {
        return AscendType::FP16;
    }
}

static at::Tensor match_rows(const at::Tensor &idx, int64_t rows);

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0), "the first dimension of x, y should be same");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    at::Tensor idx = match_rows(indices, x.size(0)); // tolerate padded x (compile/aclgraph)
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr(); 
    void* indices_ptr = idx.data_ptr();
    int indices_size = idx.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, input_hidden_token,
                          lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size, num_tokens_per_core,
                         input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y,
                       int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0), "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor idx = match_rows(indices, x.size(0));   // tolerate padded x
    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* indices_ptr = idx.data_ptr();
    int indices_size = idx.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size, lora_rank,
                          slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        bgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, y_out_ptr, batch_size,
                         num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                 at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size,
                          seq_len_ptr, seq_len_size, y_ptr,
                          batch_size, input_hidden_token, lora_rank, scale_f]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_shrink_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size,
                         y_ptr, batch_size,
                         num_tokens_per_core, input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == torch::kHalf || scalar_type == torch::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) <= slice_size, "hidden in should be smaller than hidden out");
    TORCH_CHECK(slice_offset >= 0, "slice offset should be no smaller than 0");
    TORCH_CHECK((slice_size + slice_offset) <= y.size(1),
                "slice_size + slice_offset should be smaller than the second dimension of y")

    at::Tensor y_out = y;
    void* x_ptr = x.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* lora_indices_ptr = lora_indices.data_ptr();
    void* seq_len_ptr = seq_len.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    int seq_len_size = seq_len.size(0);
    void* y_ptr = y.data_ptr();
    void* y_out_ptr = y_out.data_ptr();
    int batch_size = x.size(0);
    int lora_rank = x.size(1);
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                          batch_size, lora_rank, slice_offset, slice_size, output_full_dim]() -> int {
        auto dtype = get_dtype_from_torch(scalar_type);
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK("num_tokens_per_core != 0", "num_tokens_per_core should not be 0");
        sgmv_expand_impl(dtype, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr, seq_len_size, y_ptr, y_out_ptr,
                         batch_size, num_tokens_per_core, lora_rank, slice_size, slice_offset, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

// ---------------------------------------------------------------------------
// Fused LoRA apply: one opaque op per add_shrink/add_expand that branches on token_num. 
// prefill (token_num > threshold) -> gmm (aclnnGroupedMatmulV4); decode -> bgmv
// ---------------------------------------------------------------------------

static at::Tensor lora_grouped_matmul(const at::Tensor &x, const at::Tensor &weight,
                                      const at::Tensor &group_list,
                                      int64_t group_list_type = 1,
                                      const c10::optional<at::Tensor> &out_opt = c10::nullopt)
{
    TORCH_CHECK(group_list_type == 1,
                "lora_grouped_matmul: only group_list_type=1 (per-expert counts) is "
                "implemented, got ", group_list_type);
    int64_t T = x.size(0);
    int64_t N = weight.size(weight.dim() - 1);
    const bool use_out = out_opt.has_value() && out_opt.value().scalar_type() == x.scalar_type()
                         && out_opt.value().size(0) == T && out_opt.value().size(1) == N;
    at::Tensor out = use_out ? out_opt.value() : at::empty({T, N}, x.options());
    std::vector<at::Tensor> xv{x}, wv{weight}, ov{out};
    at::TensorList xs(xv), ws(wv), result(ov);
    at::TensorList none, act_out, dyn;
    int64_t split_item = 2, group_type = 0, act_type = 0;
    EXEC_NPU_CMD(aclnnGroupedMatmulV4, xs, ws, none, none, none, none,
                 none, none, group_list, none,
                 none, none, split_item, group_type,
                 group_list_type, act_type, result, act_out, dyn);
    return out;
}

static at::Tensor gather_weights_for_gmm(const at::Tensor &w_in, const at::Tensor &lora_indices)
{
    at::Tensor w = (w_in.dim() == 4) ? w_in.squeeze(1) : w_in;
    at::Tensor safe = lora_indices.clamp_min(0);
    at::Tensor g = w.index_select(0, safe);
    g.masked_fill_(lora_indices.lt(0).unsqueeze(1).unsqueeze(2), 0);
    return g.transpose(1, 2);
}

static at::Tensor match_rows(const at::Tensor &idx, int64_t rows)
{
    if (idx.size(0) == rows) return idx;
    if (idx.size(0) < rows) return at::constant_pad_nd(idx, {0, rows - idx.size(0)}, -1);
    return idx.slice(0, 0, rows);
}
static at::Tensor moe_bgmv_weight(const at::Tensor &w_in)
{
    auto s = w_in.sizes();
    return w_in.reshape({s[0] * s[1], s[2], s[3]});
}

static at::Tensor moe_gmm_shrink_weight(const std::vector<at::Tensor> &lora_a)
{
    const int64_t E = lora_a[0].size(1), K = lora_a[0].size(3);
    const int64_t stacked =
        static_cast<int64_t>(lora_a.size()) * lora_a[0].size(0) * lora_a[0].size(2);
    std::vector<at::Tensor> views;
    views.reserve(lora_a.size());
    for (const at::Tensor &a : lora_a) {
        views.push_back(a.permute({1, 3, 0, 2}));
    }
    return at::cat(views, 2).reshape({E, K, stacked});
}

static at::Tensor moe_gmm_expand_weight(const at::Tensor &b)
{
    const int64_t L = b.size(0), E = b.size(1), N = b.size(2), R = b.size(3);
    return b.permute({1, 0, 3, 2}).reshape({E, L * R, N});
}

static at::Tensor moe_row_slot(const at::Tensor &combined, int64_t num_experts)
{
    return at::where(combined.ge(0), combined.floor_divide(num_experts),
                     at::full_like(combined, -1));
}

static void moe_shrink_gmm(std::vector<at::Tensor> &y, const at::Tensor &x,
                           const std::vector<at::Tensor> &lora_a, const at::Tensor &combined,
                           const at::Tensor &group_list, int64_t group_list_type, double scale)
{
    const int64_t S = static_cast<int64_t>(lora_a.size());
    const int64_t L = lora_a[0].size(0), E = lora_a[0].size(1), R = lora_a[0].size(2);
    at::Tensor gw = moe_gmm_shrink_weight(lora_a);
    at::Tensor x_in = (x.scalar_type() == gw.scalar_type()) ? x : x.to(gw.scalar_type());
    at::Tensor res = lora_grouped_matmul(x_in, gw, group_list, group_list_type);
    if (scale != 1.0) {
        res.mul_(scale);
    }
    const int64_t T = res.size(0);
    at::Tensor slot = moe_row_slot(match_rows(combined, T), E);
    at::Tensor keep = slot.unsqueeze(1).eq(at::arange(L, slot.options()).unsqueeze(0));
    res.view({T, S, L, R}).masked_fill_(keep.logical_not().view({T, 1, L, 1}), 0);
    for (int64_t s = 0; s < S; ++s) {
        y[s].copy_(res.slice(1, s * L * R, (s + 1) * L * R).to(y[s].scalar_type()));
    }
}

static void moe_shrink_bgmv(std::vector<at::Tensor> &y, at::Tensor &x,
                            const std::vector<at::Tensor> &lora_a, const at::Tensor &combined,
                            double scale)
{
    const size_t S = lora_a.size();
    at::Tensor idx = match_rows(combined, x.size(0));
    y[S].zero_();
    for (size_t s = 0; s < S; ++s) {
        at::Tensor aw = moe_bgmv_weight(lora_a[s]);
        at::Tensor yb = y[S].select(0, static_cast<int64_t>(s));
        bgmv_shrink(x, aw, idx, yb, scale);
    }
}

static at::Tensor moe_gmm_slice(const at::Tensor &buf, const at::Tensor &lb,
                                const at::Tensor &group_list, int64_t group_list_type,
                                const c10::optional<at::Tensor> &out_opt = c10::nullopt)
{
    at::Tensor gw = moe_gmm_expand_weight(lb);
    at::Tensor xi = (buf.scalar_type() == gw.scalar_type()) ? buf : buf.to(gw.scalar_type());
    return lora_grouped_matmul(xi, gw, group_list, group_list_type, out_opt);
}

static void moe_expand_gmm(at::Tensor &y, const std::vector<at::Tensor> &buffers,
                           const std::vector<at::Tensor> &lora_b, const at::Tensor &group_list,
                           int64_t group_list_type, const std::vector<int64_t> &output_slices,
                           int64_t offset_start, bool add_inputs)
{
    int64_t offset = offset_start;
    for (size_t s = 0; s < lora_b.size(); ++s) {
        int64_t size = output_slices[s];
        at::Tensor res = moe_gmm_slice(buffers[s], lora_b[s], group_list, group_list_type);
        at::Tensor target = y.slice(1, offset, offset + size);
        if (add_inputs) {
            target.add_(res.to(target.scalar_type()));
        } else {
            target.copy_(res.to(target.scalar_type()));
        }
        offset += size;
    }
}

static void moe_expand_bgmv(at::Tensor &y, const std::vector<at::Tensor> &buffers,
                            const std::vector<at::Tensor> &lora_b, const at::Tensor &combined,
                            const std::vector<int64_t> &output_slices, int64_t offset_start)
{
    const size_t S = lora_b.size();
    at::Tensor idx = match_rows(combined, buffers[S].size(1));
    int64_t offset = offset_start;
    for (size_t s = 0; s < S; ++s) {
        at::Tensor bw = moe_bgmv_weight(lora_b[s]);
        at::Tensor xs = buffers[S].select(0, static_cast<int64_t>(s));
        bgmv_expand(xs, bw, idx, y, offset, output_slices[s]);
        offset += output_slices[s];
    }
}

void add_lora_shrink(std::vector<at::Tensor> y, at::Tensor x, std::vector<at::Tensor> lora_a,
                     at::Tensor lora_indices, at::Tensor seq_len, at::Tensor token_lora_indices,
                     double scale, at::Tensor use_gmm, at::Tensor no_lora,
                     bool is_moe = false, int64_t group_list_type = 1)
{
    if (no_lora.item<bool>()) {
        return;
    }
    if (is_moe) {
        TORCH_CHECK(y.size() == lora_a.size() + 1,
                    "moe add_lora_shrink: expected one [T, L*R] buffer per slice plus one "
                    "[n_slices, T, R] buffer, got ", y.size(), " for ", lora_a.size(), " slices");
        if (use_gmm.item<bool>()) {
            moe_shrink_gmm(y, x, lora_a, token_lora_indices, seq_len, group_list_type, scale);
        } else {
            moe_shrink_bgmv(y, x, lora_a, token_lora_indices, scale);
        }
        return;
    }
    if (use_gmm.item<bool>()) {  // prefill -> gmm
        // WARNING: this dense gmm branch groups rows BY SEQUENCE via
        // lora_indices (one adapter per sequence-group). MoE per-row indices
        // (combined_idx = slot * E + expert) must NOT reach this branch --
        // they would trigger a catastrophic per-row weight gather. MoE callers
        // must pass is_moe=true (stacked-GMM) or use_gmm=false (bgmv).
        // NOTE: no runtime TORCH_CHECK here -- lora_indices.max().item()
        // would force a device-to-host sync on every prefill call (~86
        // dense-layer applies per chunk), measurably degrading TTFT.
        const size_t n_slices = lora_a.size();

        std::vector<at::Tensor> gws;
        gws.reserve(n_slices);
        for (size_t s = 0; s < n_slices; ++s) {
            gws.push_back(gather_weights_for_gmm(lora_a[s], lora_indices));
        }

        for (size_t s = 0; s < n_slices; ++s) {
            TORCH_CHECK(gws[s].size(2) == gws[0].size(2) && y[s].size(1) == gws[0].size(2),
                        "add_lora_shrink: all slices must share the same rank");
        }

        if (n_slices <= 1) {
            for (size_t s = 0; s < n_slices; ++s) {
                at::Tensor gw = gws[s].contiguous();
                at::Tensor x_in = (x.scalar_type() == gw.scalar_type()) ? x : x.to(gw.scalar_type());
                at::Tensor res = lora_grouped_matmul(x_in, gw, seq_len, group_list_type);  // [T, rank]
                if (scale != 1.0) {
                    res = res.mul(scale);
                }
                y[s].add_(res.to(y[s].scalar_type()));
            }
        } else {
            at::Tensor gw = at::cat(gws, 2);
            at::Tensor x_in = (x.scalar_type() == gw.scalar_type()) ? x : x.to(gw.scalar_type());
            at::Tensor res = lora_grouped_matmul(x_in, gw, seq_len, group_list_type);  // [T, n_slices*rank]
            if (scale != 1.0) {
                res = res.mul(scale);
            }
            const int64_t rank = gws[0].size(2);
            for (size_t s = 0; s < n_slices; ++s) {
                y[s].add_(res.slice(1, s * rank, (s + 1) * rank).to(y[s].scalar_type()));
            }
        }
    } else {                      // decode -> bgmv
        at::Tensor idx = match_rows(token_lora_indices, x.size(0));
        for (size_t s = 0; s < lora_a.size(); ++s) {
            bgmv_shrink(x, lora_a[s], idx, y[s], scale);
        }
    }
}

void add_lora_expand(at::Tensor y, std::vector<at::Tensor> x, std::vector<at::Tensor> lora_b,
                     at::Tensor lora_indices, at::Tensor seq_len, at::Tensor token_lora_indices,
                     std::vector<int64_t> output_slices, int64_t offset_start, bool add_inputs,
                     at::Tensor use_gmm, at::Tensor no_lora,
                     bool is_moe = false, int64_t group_list_type = 1)
{
    if (no_lora.item<bool>()) {
        return;
    }
    if (is_moe) {
        TORCH_CHECK(x.size() == lora_b.size() + 1,
                    "moe add_lora_expand: expected one [T, L*R] buffer per slice plus one "
                    "[n_slices, T, R] buffer, got ", x.size(), " for ", lora_b.size(), " slices");
        if (use_gmm.item<bool>()) {
            moe_expand_gmm(y, x, lora_b, seq_len, group_list_type, output_slices, offset_start,
                           add_inputs);
        } else {
            moe_expand_bgmv(y, x, lora_b, token_lora_indices, output_slices, offset_start);
        }
        return;
    }
    int64_t offset = offset_start;
    if (use_gmm.item<bool>()) {  // prefill -> gmm
        for (size_t s = 0; s < lora_b.size(); ++s) {
            int64_t size = output_slices[s];
            at::Tensor gw = gather_weights_for_gmm(lora_b[s], lora_indices).contiguous();
            at::Tensor xi = (x[s].scalar_type() == gw.scalar_type()) ? x[s] : x[s].to(gw.scalar_type());
            at::Tensor res = lora_grouped_matmul(xi, gw, seq_len, group_list_type);  // [T, out]
            at::Tensor target = y.slice(1, offset, offset + size);
            if (add_inputs) {
                target.add_(res.to(target.scalar_type()));
            } else {
                target.copy_(res.to(target.scalar_type()));
            }
            offset += size;
        }
    } else {                      // decode -> bgmv (always accumulates into y)
        for (size_t s = 0; s < lora_b.size(); ++s) {
            int64_t size = output_slices[s];
            at::Tensor idx = match_rows(token_lora_indices, x[s].size(0));
            bgmv_expand(x[s], lora_b[s], idx, y, offset, size);
            offset += size;
        }
    }
}

// a version with separate lora delta
at::Tensor add_lora_expand_delta(const at::Tensor &gate_up, std::vector<at::Tensor> x,
                                 std::vector<at::Tensor> lora_b, at::Tensor seq_len,
                                 int64_t group_list_type = 1)
{
    TORCH_CHECK(x.size() == lora_b.size() + 1,
                "add_lora_expand_delta: expected one [T, L*R] buffer per slice plus one "
                "[n_slices, T, R] buffer, got ", x.size());
    int64_t total = 0;
    for (const at::Tensor &lb : lora_b) {
        total += lb.size(lb.dim() - 2);
    }
    TORCH_CHECK(total == gate_up.size(1), "add_lora_expand_delta: slices total ", total,
                " must match gate_up width ", gate_up.size(1));

    at::Tensor delta = at::empty_like(gate_up);
    int64_t offset = 0;
    for (size_t s = 0; s < lora_b.size(); ++s) {
        int64_t size = lora_b[s].size(lora_b[s].dim() - 2);
        at::Tensor view = delta.slice(1, offset, offset + size);
        at::Tensor res = moe_gmm_slice(x[s], lora_b[s], seq_len, group_list_type, view);
        if (!res.is_same(view)) {
            view.copy_(res.to(view.scalar_type()));
        }
        offset += size;
    }
    return delta;
}

static bool add_lora_eligible(const at::Tensor& y, const at::Tensor& x,
                              const std::vector<at::Tensor>& lora_a,
                              const std::vector<at::Tensor>& lora_b,
                              const std::vector<int64_t>& output_slices,
                              int64_t offset_start, const char** reason)
{
    auto fp16_or_bf16 = [](const at::Tensor& t) {
        return t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16;
    };
    *reason = nullptr;
    if (!fp16_or_bf16(x) || !fp16_or_bf16(y)) {
        *reason = "x/y dtype is not fp16/bf16";
        return false;
    }
    if (x.scalar_type() != y.scalar_type()) {
        *reason = "x and y dtypes differ";
        return false;
    }
    if (lora_a.size() > 4) {
        *reason = "more than 4 output slices (kernel supports qkv/gate_up/o_proj/qkvz)";
        return false;
    }
    if (x.size(1) % 16 != 0) {
        *reason = "x width is not a multiple of 16";
        return false;
    }
    if (y.size(1) % 16 != 0 || offset_start % 16 != 0) {
        *reason = "y width/offset is not a multiple of 16";
        return false;
    }
    if (offset_start != 0) {
        // the fused kernels address y from column 0; vllm only ever calls
        // with 0 today — reject anything else rather than silently mis-write
        *reason = "non-zero offset_start is unsupported by the fused kernel";
        return false;
    }
    for (size_t s = 0; s < lora_a.size(); ++s) {
        if (lora_a[s].scalar_type() != x.scalar_type()
            || lora_b[s].scalar_type() != x.scalar_type()) {
            *reason = "lora weight dtype differs from x";
            return false;
        }
        if (lora_a[s].size(1) != 1) {
            *reason = "weight layer dim (L) is not 1";
            return false;
        }
        int64_t r = lora_a[s].size(2);
        // z2 replicates z1 into a 64-element repeat (rank must divide 64)
        // and z1's GM workspace rows must stay 32B-aligned (rank multiple
        // of 8); the reduce tree covers exactly 8/16/32/64. Narrow ranks
        // (< 8) and non-divisors (e.g. 24, 48) stay on the bgmv/gmm
        // fallback.
        if (r != 8 && r != 16 && r != 32 && r != 64) {
            *reason = "lora rank must be one of 8/16/32/64 "
                      "(fused kernel reduce-tree shapes)";
            return false;
        }
        if (output_slices[s] % 16 != 0) {
            *reason = "output slice is not a multiple of 16";
            return false;
        }
    }
    return true;
}

constexpr int64_t kAddLoraPrefillMaxTokens = 2048;   // prefill: absolute T cap
constexpr int64_t kAddLoraPrefillMaxWork = 200000000;  // T*R*(H1+sum(H2)) cap
constexpr int64_t kAddLoraDecodeMaxTokens = 256;     // decode: fused below, bgmv above
constexpr int64_t kAddLoraDecodeMaxMergedTokens = 1024;  // decode for merged layers like gate_up_proj

// Fused LoRA apply via the in-tree split kernels (csrc/kernels/add_lora_fused.cpp):
// z1 (rank-group parallel) then z2 (token x chunk parallel over the
// concatenated slices), launched DIRECTLY on the calling thread's current
// stream. Slice descriptors (weights + widths, up to 4 — covers qkv/gate_up/o_proj
// and qwen3-style qkvz) are packed into plain arrays for the impl entry.
static void add_lora_fused_inplace(const at::Tensor& y, const at::Tensor& x,
                                   const std::vector<at::Tensor>& lora_a,
                                   const std::vector<at::Tensor>& lora_b,
                                   const at::Tensor& token_lora_indices,
                                   const std::vector<int64_t>& output_slices,
                                   int64_t offset_start, double scale, bool add_inputs)
{
    at::Tensor idx = match_rows(token_lora_indices, x.size(0));
    auto dtype = get_dtype_from_torch(x.scalar_type());
    uint32_t batch = static_cast<uint32_t>(x.size(0));
    uint32_t h1 = static_cast<uint32_t>(x.size(1));
    uint32_t r = static_cast<uint32_t>(lora_a[0].size(2));
    uint32_t y_width = static_cast<uint32_t>(y.size(1));
    float scale_f = static_cast<float>(scale);
    uint32_t add_i = add_inputs ? 1 : 0;
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    const size_t nSlices = lora_a.size();
    at::Tensor z1ws = at::empty({(int64_t)nSlices, (int64_t)batch, (int64_t)r},
                                x.options().dtype(at::kFloat));

    // vector-core count is invariant per process; querying the driver on
    // every step is wasted work
    static uint32_t aiv = [] {
        int device_id = 0;
        int64_t n = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &n)
                    == ACL_SUCCESS);
        return n > 0 ? static_cast<uint32_t>(n) : 1u;
    }();

    std::array<void *, 4> wa{};
    std::array<void *, 4> wb{};
    std::array<uint32_t, 4> h2{};
    for (size_t sl = 0; sl < nSlices; ++sl) {
        wa[sl] = lora_a[sl].data_ptr();
        wb[sl] = lora_b[sl].data_ptr();
        h2[sl] = static_cast<uint32_t>(output_slices[sl]);  // the op contract
    }
    add_lora_fused_impl(dtype, stream, x.data_ptr(), wa.data(), wb.data(),
                        idx.data_ptr(), y.data_ptr(), z1ws.data_ptr(), batch, h1, r,
                        h2.data(), static_cast<uint32_t>(nSlices), y_width, scale_f,
                        add_i, aiv);
}

// vector-core count is invariant per process (see add_lora_fused_inplace)
static uint32_t CliAivNum()
{
    static uint32_t aiv = [] {
        int device_id = 0;
        int64_t cnt = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cnt)
                    == ACL_SUCCESS);
        return cnt > 0 ? static_cast<uint32_t>(cnt) : 1u;
    }();
    return aiv;
}

// Combined LoRA gather index for the AllGather MoE backend: dest is cast to
// fp32 and argsorted by aten, then the in-tree kernel
// (csrc/kernels/combined_lora_idx.cpp) folds the per-pair slot/expert/adapter
// lookup into one gather pass. Output is a [num_pairs] int64 tensor, -1 where
// a row has no active (lora, expert) pair.
// Bit-identical to _build_combined_lora_idx_allgather_torch. The kernel only
// consumes aten-produced tensors: an aten op reading the output of a
// raw-launched kernel is not stream-ordered, so keys/argsort stay aten.
at::Tensor build_combined_lora_idx(at::Tensor dest, at::Tensor topk_ids,
                                   at::Tensor lora_indices, at::Tensor adapter_enabled,
                                   int64_t first_expert_idx, int64_t num_experts, int64_t top_k)
{
    TORCH_CHECK(dest.scalar_type() == at::kInt || dest.scalar_type() == at::kLong,
                "dest must be int32/int64");
    TORCH_CHECK(topk_ids.scalar_type() == at::kInt || topk_ids.scalar_type() == at::kLong,
                "topk_ids must be int32/int64");
    TORCH_CHECK(lora_indices.scalar_type() == at::kLong, "lora_indices must be int64");
    TORCH_CHECK(lora_indices.dim() == 1 && adapter_enabled.dim() == 1,
                "lora_indices/adapter_enabled must be 1D");
    TORCH_CHECK(top_k > 0 && num_experts > 0, "top_k/num_experts must be positive");
    const int64_t n = dest.numel();
    TORCH_CHECK(topk_ids.numel() == n, "dest and topk_ids must have equal numel");
    TORCH_CHECK(n % top_k == 0, "dest numel must be a multiple of top_k");
    if (n == 0) {
        return at::empty({0}, dest.options().dtype(at::kLong));
    }
    at::Tensor destc = dest.reshape(-1);
    at::Tensor topkc = topk_ids.reshape(-1);
    TORCH_CHECK(destc.is_contiguous() && topkc.is_contiguous(),
                "dest/topk_ids must be contiguous");
    at::Tensor lorac = lora_indices.contiguous();
    at::Tensor aec = adapter_enabled.to(at::kInt).contiguous();

    // keys: dest cast to fp32 with inactive pairs (-1) pushed beyond the
    // active range, so each active pair lands at row dest[p] and only
    // inactive pairs share the trailing n tie (their order is irrelevant:
    // every inactive pair yields -1 in the kernel, since enabled requires
    // dest >= 0). fp32 keeps the NPU sort on the vector cores (int64 sorts
    // fall back to AiCPU).
    at::Tensor keys = destc.to(at::kFloat);
    keys.masked_fill_(keys.lt(0), static_cast<float>(n));
    at::Tensor inv = at::argsort(keys);

    const uint32_t num_pairs = static_cast<uint32_t>(n);
    const uint32_t num_valid = static_cast<uint32_t>(
        std::min<int64_t>(n / top_k, lorac.numel()));
    at::Tensor out = at::empty({n}, dest.options().dtype(at::kLong));
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    turbo_ops_lora::combined_lora_idx_finalize_impl(
        stream, destc.data_ptr(), topkc.data_ptr(), lorac.data_ptr(), aec.data_ptr(),
        inv.data_ptr(), out.data_ptr(), num_pairs, static_cast<uint32_t>(top_k), num_valid,
        first_expert_idx, static_cast<uint32_t>(num_experts), CliAivNum(),
        destc.scalar_type() == at::kInt, topkc.scalar_type() == at::kInt);
    return out;
}

static constexpr int64_t kSwigluQuantMaxWidth = 4096;

std::tuple<at::Tensor, at::Tensor, at::Tensor> add_lora_swiglu_quant(
    const at::Tensor &gate_up, const c10::optional<at::Tensor> &delta, double swiglu_limit = 0.0)
{
    TORCH_CHECK(gate_up.dim() == 2, "add_lora_swiglu_quant: gate_up must be 2-D [T, 2W], got ",
                gate_up.dim(), "-D");
    TORCH_CHECK(gate_up.is_contiguous(), "add_lora_swiglu_quant: gate_up must be contiguous");
    const auto dtype = gate_up.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
                "add_lora_swiglu_quant: gate_up must be bf16 or fp16, got ", dtype);

    const int64_t batch = gate_up.size(0);
    const int64_t two_w = gate_up.size(1);
    TORCH_CHECK(two_w % 2 == 0, "add_lora_swiglu_quant: gate_up width must be even, got ", two_w);
    const int64_t width = two_w / 2;
    TORCH_CHECK(width > 0 && width <= kSwigluQuantMaxWidth, "add_lora_swiglu_quant: W must be in [1, ",
                kSwigluQuantMaxWidth, "], got ", width);
    // The kernel's W%64 tail reduce misaligns UB accesses (vector core fault
    // for tail in {8,40,48,56,...}) or miscomputes the row max (silent
    // quant corruption for tail 16/48) - bz-ascend sweep 2026-10-14.
    TORCH_CHECK(width % 64 == 0, "add_lora_swiglu_quant: W must be a multiple of 64, got ", width);

    const bool has_delta = delta.has_value();
    if (has_delta) {
        const at::Tensor &d = delta.value();
        TORCH_CHECK(d.sizes() == gate_up.sizes(), "add_lora_swiglu_quant: delta shape ", d.sizes(),
                    " must match gate_up ", gate_up.sizes());
        TORCH_CHECK(d.scalar_type() == dtype,
                    "add_lora_swiglu_quant: delta dtype ", d.scalar_type(),
                    " must match gate_up ", dtype);
        TORCH_CHECK(d.is_contiguous(), "add_lora_swiglu_quant: delta must be contiguous");
    }

    at::Tensor act = at::empty({batch, width}, gate_up.options());
    at::Tensor y = at::empty({batch, width}, gate_up.options().dtype(at::kChar));
    at::Tensor scale = at::empty({batch}, gate_up.options().dtype(at::kFloat));
    if (batch == 0) {
        return {act, y, scale};
    }

    add_lora_swiglu_quant_impl(get_dtype_from_torch(dtype),
                               c10_npu::getCurrentNPUStream().stream(), gate_up.data_ptr(),
                               has_delta ? delta.value().data_ptr() : nullptr, act.data_ptr(),
                               y.data_ptr(), scale.data_ptr(), static_cast<uint32_t>(batch),
                               static_cast<uint32_t>(width), has_delta ? 1u : 0u,
                               static_cast<float>(swiglu_limit), CliAivNum());
    return {act, y, scale};
}

// Same layout contract as add_lora_shrink/add_lora_expand. Branches in C++ on
// the CPU flag tensors (aclgraph-safe, opaque to torch.compile):
//   no_lora            -> skip
//   use_gmm (prefill)  -> in-tree fused kernel below kAddLoraPrefillMaxTokens,
//                         gmm/GroupedMatmul above
//   decode             -> in-tree fused kernel below kAddLoraDecodeMaxTokens,
//                         bgmv shrink+expand pair above — kept
// The fused kernel (csrc/kernels/add_lora_fused.cpp) is native fp16/bf16 with
// fp32 accumulation.
void add_lora(at::Tensor y, at::Tensor x, std::vector<at::Tensor> lora_a,
              std::vector<at::Tensor> lora_b, at::Tensor lora_indices, at::Tensor seq_len,
              at::Tensor token_lora_indices, std::vector<int64_t> output_slices,
              int64_t offset_start, double scale, bool add_inputs,
              at::Tensor use_gmm, at::Tensor no_lora, at::Tensor use_add_lora)
{
    if (no_lora.item<bool>() || x.size(0) == 0) {
        return;
    }
    TORCH_CHECK(lora_a.size() == lora_b.size() && lora_a.size() == output_slices.size(),
                "add_lora: lora_a (", lora_a.size(), "), lora_b (", lora_b.size(),
                ") and output_slices (", output_slices.size(), ") must have equal sizes");
    const bool prefill = use_gmm.item<bool>();
    bool size_ok;
    if (prefill) {
        int64_t sum_h2 = 0;
        int64_t rank = lora_a.empty() ? 0 : lora_a[0].size(2);
        for (int64_t sz : output_slices) {
            sum_h2 += sz;
        }
        const int64_t work = x.size(0) * rank * (x.size(1) + sum_h2);
        size_ok = x.size(0) <= kAddLoraPrefillMaxTokens && work <= kAddLoraPrefillMaxWork;
    } else {
        size_ok = x.size(0) <= kAddLoraDecodeMaxTokens
                  || (lora_a.size() >= 2 && x.size(0) <= kAddLoraDecodeMaxMergedTokens);
    }
    const bool want_fused = use_add_lora.item<bool>() && size_ok;
    const char* reason = nullptr;
    bool eligible = want_fused &&
        add_lora_eligible(y, x, lora_a, lora_b, output_slices, offset_start, &reason);
    if (eligible) {
        add_lora_fused_inplace(y, x, lora_a, lora_b, token_lora_indices, output_slices,
                               offset_start, scale, add_inputs);
        return;
    }
    if (reason != nullptr) {
        TORCH_WARN_ONCE("add_lora fused kernel not eligible (", reason, "); falling back to the ",
                        prefill ? "gmm" : "bgmv", " path");
    }
    int64_t offset = offset_start;
    // Fallback: delegate to add_lora_shrink + add_lora_expand (dense path,
    // is_moe=false) instead of duplicating their gmm/bgmv logic here. The
    // shrink buffers are fp32 (bgmv writes fp32; the gmm path upcasts the
    // bf16 result, matching the historical behavior of this fallback).
    {
        const size_t n_slices = lora_a.size();
        const int64_t rank = n_slices ? lora_a[0].size(2) : 0;
        // One zero buffer for all slices; select(0, s) keeps each view
        // contiguous so bgmv can consume it directly. Zero-fill is required
        // for -1 (no-lora) rows, which bgmv skips.
        at::Tensor bufs = at::zeros({static_cast<int64_t>(n_slices), x.size(0), rank},
                                    x.options().dtype(at::kFloat));
        std::vector<at::Tensor> shrink_outputs;
        shrink_outputs.reserve(n_slices);
        for (size_t s = 0; s < n_slices; ++s) {
            shrink_outputs.push_back(bufs.select(0, static_cast<int64_t>(s)));
        }
        add_lora_shrink(shrink_outputs, x, lora_a, lora_indices, seq_len,
                        token_lora_indices, scale, use_gmm, no_lora);
        add_lora_expand(y, shrink_outputs, lora_b, lora_indices, seq_len,
                        token_lora_indices, output_slices, offset_start, add_inputs,
                        use_gmm, no_lora);
    }
}


}  // namespace turbo_ops_lora

TORCH_LIBRARY_FRAGMENT(turbo_ops, ops) {
    // Direct kernel custom ops
    ops.def("bgmv_shrink(Tensor x, Tensor weight, Tensor indices, Tensor(a!) y, float scale) -> ()");
    ops.impl("bgmv_shrink", torch::kPrivateUse1, &turbo_ops_lora::bgmv_shrink);

    ops.def(
        "bgmv_expand(Tensor x, Tensor weight, Tensor indices, Tensor(a!) y,"
        "            int slice_offset, int slice_size) -> Tensor(a!)");
    ops.impl("bgmv_expand", torch::kPrivateUse1, &turbo_ops_lora::bgmv_expand);

    ops.def("sgmv_shrink(Tensor x, Tensor weight, Tensor lora_indices, Tensor seq_len, Tensor(a!) y, float scale) -> ()");
    ops.impl("sgmv_shrink", torch::kPrivateUse1, &turbo_ops_lora::sgmv_shrink);

    ops.def(
        "sgmv_expand(Tensor x, Tensor weight, Tensor lora_indices, Tensor seq_len, Tensor(a!) y,"
        "            int slice_offset, int slice_size) -> Tensor(a!)");
    ops.impl("sgmv_expand", torch::kPrivateUse1, &turbo_ops_lora::sgmv_expand);

    ops.def(
        "add_lora_shrink(Tensor(a!)[] y, Tensor x, Tensor[] lora_a, Tensor lora_indices,"
        "                Tensor seq_len, Tensor token_lora_indices, float scale,"
        "                Tensor use_gmm, Tensor no_lora, bool is_moe=False,"
        "                int group_list_type=1) -> ()");
    ops.impl("add_lora_shrink", torch::kPrivateUse1, &turbo_ops_lora::add_lora_shrink);

    ops.def(
        "add_lora_expand(Tensor(a!) y, Tensor[] x, Tensor[] lora_b, Tensor lora_indices,"
        "                Tensor seq_len, Tensor token_lora_indices, int[] output_slices,"
        "                int offset_start, bool add_inputs, Tensor use_gmm, Tensor no_lora,"
        "                bool is_moe=False, int group_list_type=1) -> ()");
    ops.impl("add_lora_expand", torch::kPrivateUse1, &turbo_ops_lora::add_lora_expand);

    // a version with separate lora delta
    ops.def(
        "add_lora_expand_delta(Tensor gate_up, Tensor[] x, Tensor[] lora_b, Tensor seq_len,"
        "                      int group_list_type=1) -> Tensor");
    ops.impl("add_lora_expand_delta", torch::kPrivateUse1, &turbo_ops_lora::add_lora_expand_delta);

    // fused w13 epilogue: delta add + swiglu + per-token int8 quant
    ops.def(
        "add_lora_swiglu_quant(Tensor gate_up, Tensor? delta=None, float swiglu_limit=0.0)"
        " -> (Tensor act, Tensor y, Tensor scale)");
    ops.impl("add_lora_swiglu_quant", torch::kPrivateUse1, &turbo_ops_lora::add_lora_swiglu_quant);

    // fused LoRA apply for add_lora_linear: in-tree fused kernel / gmm / bgmv
    ops.def(
        "add_lora(Tensor(a!) y, Tensor x, Tensor[] lora_a, Tensor[] lora_b,"
        "          Tensor lora_indices, Tensor seq_len, Tensor token_lora_indices,"
        "          int[] output_slices, int offset_start, float scale, bool add_inputs,"
        "          Tensor use_gmm, Tensor no_lora, Tensor use_add_lora) -> ()");
    ops.impl("add_lora", torch::kPrivateUse1, &turbo_ops_lora::add_lora);

    // combined (lora, expert) gather index for the AllGather MoE-LoRA path
    ops.def(
        "build_combined_lora_idx(Tensor dest, Tensor topk_ids, Tensor lora_indices,"
        "                       Tensor adapter_enabled, int first_expert_idx,"
        "                       int num_experts, int top_k) -> Tensor");
    ops.impl("build_combined_lora_idx", torch::kPrivateUse1, &turbo_ops_lora::build_combined_lora_idx);

}

// The module contains dispatcher registrations only; no Python dependency on vLLM.
PyMODINIT_FUNC PyInit__C() {
    static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
    return PyModule_Create(&module);
}
