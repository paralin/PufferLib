// cuDNN Conv2d: forward/backward with fused bias+activation.
// Included by ocean.cu (training) and tests/test_nmmo3_cuda.cu (test).

#ifndef CUDNN_CONV2D_CU
#define CUDNN_CONV2D_CU

#include <hip/hip_runtime.h>
#include <hipDNN.h>
#include <cstdio>

#include "kernels.hip.cpp"

#ifndef CHECK_CUDNN
#define CHECK_CUDNN(call) do { \
    hipdnnStatus_t e = call; \
    if (e != HIPDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN %s:%d: %s\n", __FILE__, __LINE__, hipdnnGetErrorString(e)); exit(1); \
    } \
} while(0)
#endif

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

static hipdnnHandle_t get_cudnn_handle() {
    static hipdnnHandle_t h = nullptr;
    if (!h) CHECK_CUDNN(hipdnnCreate(&h));
    return h;
}

// ---- ConvWeights: params + batch-independent cuDNN state ----

struct ConvWeights {
    PrecisionTensor w, b;  // w: (OC, IC*K*K), b: (OC)
    int IC, OC, K, S, IH, IW, OH, OW;
    bool relu;
    hipdnnDataType_t dtype;
    hipdnnTensorDescriptor_t cudnn_bias;
    hipdnnFilterDescriptor_t cudnn_filt;
    hipdnnConvolutionDescriptor_t cudnn_conv;
    hipdnnActivationDescriptor_t cudnn_act;
    bool cudnn_ready;
};

// ---- ConvActivations: per-batch-size buffers + descriptors ----

struct ConvActivations {
    PrecisionTensor out, grad, saved_input;
    PrecisionTensor wgrad, bgrad;
    // Per-batch-size cuDNN state
    hipdnnTensorDescriptor_t cudnn_in, cudnn_out;
    hipdnnConvolutionFwdAlgo_t fwd_algo;
    hipdnnConvolutionBwdDataAlgo_t bwd_data_algo;
    hipdnnConvolutionBwdFilterAlgo_t bwd_filt_algo;
    size_t fwd_ws_bytes, bwd_data_ws_bytes, bwd_filt_ws_bytes;
    void* fwd_ws; void* bwd_data_ws; void* bwd_filt_ws;
    bool cudnn_setup;
};

static void conv_init(ConvWeights* cw, int IC, int OC, int K, int S, int IH, int IW, bool relu) {
    cw->IC = IC; cw->OC = OC; cw->K = K; cw->S = S; cw->IH = IH; cw->IW = IW;
    cw->OH = (IH - K) / S + 1; cw->OW = (IW - K) / S + 1;
    cw->relu = relu; cw->cudnn_ready = false;
}

// Create batch-independent descriptors (once)
static void conv_setup_shared(ConvWeights* cw, hipdnnDataType_t dt) {
    if (cw->cudnn_ready) return;
    cw->dtype = dt;
    CHECK_CUDNN(hipdnnCreateFilterDescriptor(&cw->cudnn_filt));
    CHECK_CUDNN(hipdnnSetFilter4dDescriptor(cw->cudnn_filt, dt, HIPDNN_TENSOR_NCHW, cw->OC, cw->IC, cw->K, cw->K));
    CHECK_CUDNN(hipdnnCreateConvolutionDescriptor(&cw->cudnn_conv));
    CHECK_CUDNN(hipdnnSetConvolution2dDescriptor(cw->cudnn_conv, 0, 0, cw->S, cw->S, 1, 1, HIPDNN_CROSS_CORRELATION, HIPDNN_DATA_FLOAT));
    CHECK_CUDNN(hipdnnCreateTensorDescriptor(&cw->cudnn_bias));
    CHECK_CUDNN(hipdnnSetTensor4dDescriptor(cw->cudnn_bias, HIPDNN_TENSOR_NCHW, dt, 1, cw->OC, 1, 1));
    CHECK_CUDNN(hipdnnCreateActivationDescriptor(&cw->cudnn_act));
    CHECK_CUDNN(hipdnnSetActivationDescriptor(cw->cudnn_act,
        cw->relu ? HIPDNN_ACTIVATION_RELU : HIPDNN_ACTIVATION_PATHTRU, HIPDNN_NOT_PROPAGATE_NAN, 0.0));
    cw->cudnn_ready = true;
}

// Setup per-activation-set cuDNN state: batch-dependent descriptors + algo search + workspace
static void conv_setup_activations(ConvWeights* cw, ConvActivations* ca, int B, hipdnnDataType_t dt) {
    conv_setup_shared(cw, dt);
    hipdnnHandle_t h = get_cudnn_handle();

    CHECK_CUDNN(hipdnnCreateTensorDescriptor(&ca->cudnn_in));
    CHECK_CUDNN(hipdnnSetTensor4dDescriptor(ca->cudnn_in, HIPDNN_TENSOR_NCHW, dt, B, cw->IC, cw->IH, cw->IW));
    CHECK_CUDNN(hipdnnCreateTensorDescriptor(&ca->cudnn_out));
    CHECK_CUDNN(hipdnnSetTensor4dDescriptor(ca->cudnn_out, HIPDNN_TENSOR_NCHW, dt, B, cw->OC, cw->OH, cw->OW));

    int returned;
    hipdnnConvolutionFwdAlgoPerf_t fp;
    CHECK_CUDNN(hipdnnFindConvolutionForwardAlgorithm(h, ca->cudnn_in, cw->cudnn_filt, cw->cudnn_conv, ca->cudnn_out, 1, &returned, &fp));
    ca->fwd_algo = fp.algo;
    CHECK_CUDNN(hipdnnGetConvolutionForwardWorkspaceSize(h, ca->cudnn_in, cw->cudnn_filt, cw->cudnn_conv, ca->cudnn_out, ca->fwd_algo, &ca->fwd_ws_bytes));
    ca->fwd_ws = NULL; if (ca->fwd_ws_bytes > 0) hipMalloc(&ca->fwd_ws, ca->fwd_ws_bytes);

    hipdnnConvolutionBwdFilterAlgoPerf_t ffp;
    CHECK_CUDNN(hipdnnFindConvolutionBackwardFilterAlgorithm(h, ca->cudnn_in, ca->cudnn_out, cw->cudnn_conv, cw->cudnn_filt, 1, &returned, &ffp));
    ca->bwd_filt_algo = ffp.algo;
    CHECK_CUDNN(hipdnnGetConvolutionBackwardFilterWorkspaceSize(h, ca->cudnn_in, ca->cudnn_out, cw->cudnn_conv, cw->cudnn_filt, ca->bwd_filt_algo, &ca->bwd_filt_ws_bytes));
    ca->bwd_filt_ws = NULL; if (ca->bwd_filt_ws_bytes > 0) hipMalloc(&ca->bwd_filt_ws, ca->bwd_filt_ws_bytes);

    hipdnnConvolutionBwdDataAlgoPerf_t dp;
    CHECK_CUDNN(hipdnnFindConvolutionBackwardDataAlgorithm(h, cw->cudnn_filt, ca->cudnn_out, cw->cudnn_conv, ca->cudnn_in, 1, &returned, &dp));
    ca->bwd_data_algo = dp.algo;
    CHECK_CUDNN(hipdnnGetConvolutionBackwardDataWorkspaceSize(h, cw->cudnn_filt, ca->cudnn_out, cw->cudnn_conv, ca->cudnn_in, ca->bwd_data_algo, &ca->bwd_data_ws_bytes));
    ca->bwd_data_ws = NULL; if (ca->bwd_data_ws_bytes > 0) hipMalloc(&ca->bwd_data_ws, ca->bwd_data_ws_bytes);

    ca->cudnn_setup = true;
}

// Legacy single-setup API (for tests)
static void conv_setup(ConvWeights* cw, int B, hipdnnDataType_t dt) {
    conv_setup_shared(cw, dt);
}

static void conv_reg_params(ConvWeights* cw, Allocator* alloc) {
    cw->w = {.shape = {cw->OC, cw->IC * cw->K * cw->K}};
    cw->b = {.shape = {cw->OC}};
    alloc_register(alloc,&cw->w); alloc_register(alloc,&cw->b);
}

static void conv_reg_train(ConvWeights* cw, ConvActivations* ca, Allocator* acts, Allocator* grads, int B, hipdnnDataType_t dt) {
    ca->out         = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->grad        = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->saved_input = {.shape = {B * cw->IC * cw->IH * cw->IW}};
    ca->wgrad       = {.shape = {cw->OC, cw->IC * cw->K * cw->K}};
    ca->bgrad       = {.shape = {cw->OC}};
    alloc_register(acts,&ca->out); alloc_register(acts,&ca->grad); alloc_register(acts,&ca->saved_input);
    alloc_register(grads,&ca->wgrad); alloc_register(grads,&ca->bgrad);
    conv_setup_activations(cw, ca, B, dt);
}

static void conv_reg_rollout(ConvWeights* cw, ConvActivations* ca, Allocator* alloc, int B, hipdnnDataType_t dt) {
    ca->out = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->cudnn_setup = false;
    alloc_register(alloc,&ca->out);
    conv_setup_activations(cw, ca, B, dt);
}

static void conv_init_weights(ConvWeights* cw, uint64_t* seed, hipStream_t stream) {
    PrecisionTensor wt = {.data = cw->w.data, .shape = {cw->OC, cw->IC * cw->K * cw->K}};
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
    hipMemsetAsync(cw->b.data, 0, numel(cw->b.shape) * sizeof(precision_t), stream);
}

// ---- Forward / Backward ----

// Fused conv + bias + activation. All NCHW. Saves input for backward.
static void conv_forward(ConvWeights* cw, ConvActivations* ca, void* input, int B, hipStream_t stream) {
    hipdnnHandle_t h = get_cudnn_handle();
    CHECK_CUDNN(hipdnnSetStream(h, stream));
    float alpha = 1.0f, beta = 0.0f;
    if (ca->saved_input.data) {
        hipMemcpyAsync(ca->saved_input.data, input,
            (int64_t)B * cw->IC * cw->IH * cw->IW * sizeof(precision_t), hipMemcpyDeviceToDevice, stream);
    }
    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(h,
        &alpha, ca->cudnn_in, input, cw->cudnn_filt, cw->w.data,
        cw->cudnn_conv, ca->fwd_algo, ca->fwd_ws, ca->fwd_ws_bytes,
        &beta, ca->cudnn_out, ca->out.data, cw->cudnn_bias, cw->b.data,
        cw->cudnn_act, ca->cudnn_out, ca->out.data));
}

// Backward: upstream grad in ca->grad, relu mask in ca->out.
// Caller must apply relu backward and bias grad (dtype-specific kernels).
// This does cuDNN filter grad + optional data grad.
static void conv_backward(ConvWeights* cw, ConvActivations* ca, void* input_grad, int B, hipStream_t stream) {
    hipdnnHandle_t h = get_cudnn_handle();
    CHECK_CUDNN(hipdnnSetStream(h, stream));
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(hipdnnConvolutionBackwardFilter(h,
        &alpha, ca->cudnn_in, ca->saved_input.data, ca->cudnn_out, ca->grad.data,
        cw->cudnn_conv, ca->bwd_filt_algo, ca->bwd_filt_ws, ca->bwd_filt_ws_bytes,
        &beta, cw->cudnn_filt, ca->wgrad.data));

    if (input_grad) {
        CHECK_CUDNN(hipdnnConvolutionBackwardData(h,
            &alpha, cw->cudnn_filt, cw->w.data, ca->cudnn_out, ca->grad.data,
            cw->cudnn_conv, ca->bwd_data_algo, ca->bwd_data_ws, ca->bwd_data_ws_bytes,
            &beta, ca->cudnn_in, input_grad));
    }
}

#endif // CUDNN_CONV2D_CU
