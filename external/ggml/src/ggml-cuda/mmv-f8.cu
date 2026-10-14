#include "mmv-f8.cuh"
#include "mmid.cuh"

// One warp per dst row, MMV_F8_NWARPS rows per CUDA block. Each lane reads 4 of the 128 FP8 bytes of a block
// (one coalesced 128 byte load per warp; blocks are 132 bytes and therefore only 4 byte aligned) and the matching
// 4 activations as a float4.
#define MMV_F8_NWARPS 4
#define MMV_F8_LANES  (QK_F8_E4M3_B128/4)

static_assert(MMV_F8_LANES == 32, "a block must map to 32 lanes");

template <int ncols_dst, bool has_ids>
static __global__ void __launch_bounds__(MMV_F8_NWARPS*MMV_F8_LANES) mul_mat_vec_f8_e4m3_b128(
        const block_f8_e4m3_b128 * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int nblocks, const int nrows, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const uint3 sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const uint3 nchannels_y, const int64_t ids_stride) {
    const int row  = blockIdx.x*MMV_F8_NWARPS + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= nrows) {
        return;
    }

    // For MUL_MAT_ID blockIdx.y is the expert slot and blockIdx.z the token.
    const int channel_dst = blockIdx.y;
    int channel_x;
    int channel_y;
    int sample_dst;
    if constexpr (has_ids) {
        const int token = blockIdx.z;
        channel_x  = ids[channel_dst + token*ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
        y   += token*stride_col_y;
        dst += token*stride_col_dst;
    } else {
        channel_x  = fastdiv(channel_dst, channel_ratio);
        channel_y  = channel_dst;
        sample_dst = blockIdx.z;
    }
    const int sample_x = fastdiv(sample_dst, sample_ratio);

    x   += sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row;
    y   += sample_dst*stride_sample_y + channel_y*stride_channel_y;
    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;

    float sumf[ncols_dst] = {0.0f};

#pragma unroll 4
    for (int ib = 0; ib < nblocks; ++ib) {
        float4 xv;
        ggml_cuda_e4m3x4_to_fp32_scaled(reinterpret_cast<const uint32_t *>(x[ib].qs)[lane], xv);
        // The decoded values are scaled by 2^-8, fold the 256 into the block scale.
        const float d = x[ib].d * 256.0f;

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float4 yv = reinterpret_cast<const float4 *>(y + j*stride_col_y + ib*QK_F8_E4M3_B128)[lane];
            float sum = xv.x*yv.x;
            sum = fmaf(xv.y, yv.y, sum);
            sum = fmaf(xv.z, yv.z, sum);
            sum = fmaf(xv.w, yv.w, sum);
            sumf[j] = fmaf(sum, d, sumf[j]);
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<MMV_F8_LANES>(sumf[j]);
    }

    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j*stride_col_dst + row] = sumf[j];
        }
    }
}

template <int ncols_dst, bool has_ids>
static void launch_mul_mat_vec_f8_e4m3_b128(
        const block_f8_e4m3_b128 * x, const float * y, const int32_t * ids, float * dst,
        const int64_t nblocks, const int64_t nrows, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t nsamples_x, const int64_t nsamples_dst,
        const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nblocks_z, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(has_ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(nsamples_dst % nsamples_x == 0);
    const uint3 nchannels_y_fd   = has_ids ? init_fastdiv_values(nchannels_y)                 : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = has_ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst / nsamples_x);

    const dim3 block_nums((nrows + MMV_F8_NWARPS - 1) / MMV_F8_NWARPS, nchannels_dst, nblocks_z);
    const dim3 block_dims(MMV_F8_LANES, MMV_F8_NWARPS, 1);
    mul_mat_vec_f8_e4m3_b128<ncols_dst, has_ids><<<block_nums, block_dims, 0, stream>>>(
        x, y, ids, dst, nblocks, nrows, stride_row, stride_col_y, stride_col_dst,
        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_y_fd, ids_stride);
}

// MUL_MAT_ID for larger batches: blockIdx.y is the expert, whose src1/dst rows come from ggml_cuda_launch_mm_ids_helper.
// Each expert's weights are read from DRAM once. A warp computes MMV_F8_GROUPED_ROWS rows for up to
// MMV_F8_MAX_BATCH_SIZE columns at a time so that every activation load feeds several rows.
#define MMV_F8_GROUPED_ROWS 8

static __global__ void __launch_bounds__(MMV_F8_NWARPS*MMV_F8_LANES) mul_mat_vec_f8_e4m3_b128_grouped(
        const block_f8_e4m3_b128 * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids_src1,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        const int nblocks, const int nrows, const int64_t stride_row, const int64_t stride_channel_x,
        const int64_t stride_row_y, const int64_t stride_row_dst) {
    constexpr int ncols_max = MMV_F8_MAX_BATCH_SIZE;
    constexpr int nr        = MMV_F8_GROUPED_ROWS;
    const int row0 = (blockIdx.x*MMV_F8_NWARPS + threadIdx.y)*nr;
    const int lane = threadIdx.x;
    if (row0 >= nrows) {
        return;
    }

    const int expert = blockIdx.y;
    const int i_end  = expert_bounds[expert + 1];

    const block_f8_e4m3_b128 * x_rows[nr];
#pragma unroll
    for (int r = 0; r < nr; ++r) {
        x_rows[r] = x + expert*stride_channel_x + min(row0 + r, nrows - 1)*stride_row;
    }

    for (int i0 = expert_bounds[expert]; i0 < i_end; i0 += ncols_max) {
        const int ncols = min(ncols_max, i_end - i0);
        const float * y_cols[ncols_max];
#pragma unroll
        for (int j = 0; j < ncols_max; ++j) {
            y_cols[j] = y + ids_src1[i0 + min(j, ncols - 1)]*stride_row_y;
        }

        float sumf[nr][ncols_max] = {{0.0f}};
        for (int ib = 0; ib < nblocks; ++ib) {
            float4 xv[nr];
            float  d[nr];
#pragma unroll
            for (int r = 0; r < nr; ++r) {
                ggml_cuda_e4m3x4_to_fp32_scaled(reinterpret_cast<const uint32_t *>(x_rows[r][ib].qs)[lane], xv[r]);
                d[r] = x_rows[r][ib].d * 256.0f;
            }

#pragma unroll
            for (int j = 0; j < ncols_max; ++j) {
                if (j >= ncols) {
                    break;
                }
                const float4 yv = reinterpret_cast<const float4 *>(y_cols[j] + ib*QK_F8_E4M3_B128)[lane];
#pragma unroll
                for (int r = 0; r < nr; ++r) {
                    float sum = xv[r].x*yv.x;
                    sum = fmaf(xv[r].y, yv.y, sum);
                    sum = fmaf(xv[r].z, yv.z, sum);
                    sum = fmaf(xv[r].w, yv.w, sum);
                    sumf[r][j] = fmaf(sum, d[r], sumf[r][j]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < nr; ++r) {
#pragma unroll
            for (int j = 0; j < ncols_max; ++j) {
                sumf[r][j] = warp_reduce_sum<MMV_F8_LANES>(sumf[r][j]);
            }
        }

        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < ncols_max; ++j) {
                if (j >= ncols) {
                    break;
                }
                float * dst_col = dst + ids_dst[i0 + j]*stride_row_dst;
#pragma unroll
                for (int r = 0; r < nr; ++r) {
                    if (row0 + r < nrows) {
                        dst_col[row0 + r] = sumf[r][j];
                    }
                }
            }
        }
    }
}

void ggml_cuda_mul_mat_vec_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3_B128);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne13 == ne3);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const int64_t s01 = nb01 / ts_src0;
    const int64_t s02 = nb02 / ts_src0;
    const int64_t s03 = nb03 / ts_src0;
    const int64_t s11 = nb11 / ts_src1;
    const int64_t s12 = nb12 / ts_src1;
    const int64_t s13 = nb13 / ts_src1;
    const int64_t s1  =  nb1 / ts_dst;
    const int64_t s2  =  nb2 / ts_dst;
    const int64_t s3  =  nb3 / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    const block_f8_e4m3_b128 * src0_d = (const block_f8_e4m3_b128 *) src0->data;
    const float              * src1_d = (const float *) src1->data;
    const int32_t            *  ids_d = ids ? (const int32_t *) ids->data : nullptr;
    float                    *  dst_d = (float *) dst->data;

    const int64_t nblocks = ne00 / QK_F8_E4M3_B128;
    cudaStream_t stream = ctx.stream();

    if (ids && ncols_dst > MMV_F8_MMID_PER_TOKEN_MAX) {
        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_rows        = ncols_dst*n_expert_used;

        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
        ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ncols_dst, n_expert_used, ne11, ids_stride, s12/s11, stream);

        constexpr int rows_per_block = MMV_F8_NWARPS*MMV_F8_GROUPED_ROWS;
        const dim3 block_nums((ne01 + rows_per_block - 1) / rows_per_block, ne02, 1);
        const dim3 block_dims(MMV_F8_LANES, MMV_F8_NWARPS, 1);
        mul_mat_vec_f8_e4m3_b128_grouped<<<block_nums, block_dims, 0, stream>>>(
            src0_d, src1_d, ids_src1.get(), ids_dst.get(), expert_bounds.get(), dst_d, nblocks, ne01, s01, s02, s11, s1);
        return;
    }

    GGML_ASSERT(ncols_dst <= MMV_F8_MAX_BATCH_SIZE);

#define LAUNCH_MMV_F8(ncols, has_ids, nblocks_z)                                                              \
    launch_mul_mat_vec_f8_e4m3_b128<ncols, has_ids>(src0_d, src1_d, ids_d, dst_d, nblocks, ne01, s01,          \
        stride_col_y, stride_col_dst, ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst, \
        ne03, ne3, s03, s13, s3, nblocks_z, ids_stride, stream)

    if (ids) {
        // One token per blockIdx.z, so the kernel handles a single dst column.
        LAUNCH_MMV_F8(1, true, ncols_dst);
        return;
    }

    switch (ncols_dst) {
        case 1: LAUNCH_MMV_F8(1, false, ne3); break;
        case 2: LAUNCH_MMV_F8(2, false, ne3); break;
        case 3: LAUNCH_MMV_F8(3, false, ne3); break;
        case 4: LAUNCH_MMV_F8(4, false, ne3); break;
        case 5: LAUNCH_MMV_F8(5, false, ne3); break;
        case 6: LAUNCH_MMV_F8(6, false, ne3); break;
        case 7: LAUNCH_MMV_F8(7, false, ne3); break;
        case 8: LAUNCH_MMV_F8(8, false, ne3); break;
        default: GGML_ABORT("fatal error");
    }

#undef LAUNCH_MMV_F8
}

bool ggml_cuda_should_use_mmv_f8(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F8_E4M3_B128 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const size_t ts_src0 = ggml_type_size(src0->type);
    if (src0->nb[0] != ts_src0 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0->nb[i] % ts_src0 != 0) {
            return false;
        }
        // Activations are read as float4.
        if (src1->nb[i] % sizeof(float4) != 0) {
            return false;
        }
    }
    if ((uintptr_t) src1->data % sizeof(float4) != 0) {
        return false;
    }

    if (!ids) {
        return src1->ne[1] <= MMV_F8_MAX_BATCH_SIZE;
    }
    const int64_t n_tokens = dst->ne[2];
    if (n_tokens <= MMV_F8_MMID_PER_TOKEN_MAX) {
        return true;
    }
    // The grouped kernel addresses src1/dst rows through flat row indices.
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    return ids->nb[0] == sizeof(int32_t) && src1->nb[2] % src1->nb[1] == 0 && dst->nb[2] == dst->ne[1]*dst->nb[1] &&
        n_tokens*sizeof(int32_t) <= smpbo;
}
