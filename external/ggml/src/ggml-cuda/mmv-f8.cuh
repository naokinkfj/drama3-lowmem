#include "common.cuh"

#define MMV_F8_MAX_BATCH_SIZE     8 // Max. number of src1 columns for the F8_E4M3_B128 vector kernel.
#define MMV_F8_MMID_PER_TOKEN_MAX 8 // MUL_MAT_ID with up to this many tokens runs per token, larger batches per expert.

// F8_E4M3_B128 weights decoded on the fly, F32 activations and F32 accumulation.
void ggml_cuda_mul_mat_vec_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

// ids is nullptr for MUL_MAT. MUL_MAT_ID is supported for any number of tokens and never synchronizes.
bool ggml_cuda_should_use_mmv_f8(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);
