// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// LocateAnything-3B CPU runtime.
//
// This example intentionally depends only on ncnn + simpleocv.  The exported
// ncnn Qwen decoder shards are seq=1 validation graphs, so dynamic prefill,
// RoPE, GQA attention and KV cache are implemented here over raw fp16 weights
// from locate_everything/raw_runtime/weights_f16.

#include "layer.h"
#include "net.h"
#include "simpleocv.h"
#include "gpu.h"

#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

static const int token_endoftext = 151643;
static const int token_im_start = 151644;
static const int token_im_end = 151645;
static const int token_img_context = 151665;
static const int token_img_start = 151666;
static const int token_img_end = 151667;
static const int token_box_start = 151668;
static const int token_box_end = 151669;
static const int token_ref_start = 151672;
static const int token_ref_end = 151673;
static const int token_coord_start = 151677;
static const int token_coord_end = 152677;
static const int token_none = 4064;
static const int token_text_mask = 151676;
static const int token_null = 152678;
static const int token_switch = 152679;
static const int locate_block_size = 6;

static const int locate_vision_hidden_size = 1152;
static const int locate_patch_size = 14;
static const int locate_merge_h = 2;
static const int locate_merge_w = 2;
static const int moonvit_layers = 27;
static const int moonvit_heads = 16;
static const int moonvit_head_dim = 72;
static const int moonvit_mlp_hidden = 4304;
static const int moonvit_pos_grid = 64;
static const int locate_image_token_limit = 25600;
static const int locate_default_cpu_patch_limit = 64;

static const int qwen_hidden = 2048;
static const int qwen_layers = 36;
static const int qwen_heads = 16;
static const int qwen_kv_heads = 2;
static const int qwen_head_dim = 128;
static const int qwen_intermediate = 11008;
static const int qwen_vocab = 152681;
static const float qwen_rope_theta = 1000000.f;
static const float qwen_rms_eps = 1e-6f;

static bool locate_env_flag(const char* name, bool default_value)
{
    const char* v = getenv(name);
    if (!v || !v[0])
        return default_value;
    if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0 || strcmp(v, "off") == 0 || strcmp(v, "OFF") == 0)
        return false;
    return true;
}

static bool locate_vulkan_enabled()
{
#if NCNN_VULKAN
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized)
    {
        initialized = 1;
        enabled = locate_env_flag("LOCATEANYTHING_VULKAN", true);
        if (enabled)
        {
            const int gpu_count = ncnn::get_gpu_count();
            if (gpu_count <= 0)
            {
                fprintf(stderr, "locate_anything: Vulkan requested but no ncnn Vulkan device found, fallback to CPU\n");
                enabled = false;
            }
            else
            {
                const int gpu_index = ncnn::get_default_gpu_index();
                const ncnn::GpuInfo& info = ncnn::get_gpu_info(gpu_index);
                fprintf(stderr, "locate_anything: Vulkan enabled device=%d name=%s\n", gpu_index, info.device_name());
            }
        }
        else
        {
            fprintf(stderr, "locate_anything: Vulkan disabled by LOCATEANYTHING_VULKAN=0\n");
        }
    }
    return enabled;
#else
    (void)locate_env_flag;
    return false;
#endif
}

static void configure_net_runtime(ncnn::Net& net, bool prefer_vulkan)
{
    net.opt.lightmode = true;
    net.opt.use_packing_layout = true;
#if NCNN_VULKAN
    const bool use_vulkan = prefer_vulkan && locate_vulkan_enabled();
    net.opt.use_vulkan_compute = use_vulkan;
    if (use_vulkan)
    {
        const int gpu_index = ncnn::get_default_gpu_index();
        net.set_vulkan_device(gpu_index);
        const bool use_fp16 = locate_env_flag("LOCATEANYTHING_VULKAN_FP16", false);
        net.opt.use_fp16_packed = use_fp16;
        net.opt.use_fp16_storage = use_fp16;
        net.opt.use_fp16_arithmetic = use_fp16;
        net.opt.use_bf16_storage = false;
    }
#else
    (void)prefer_vulkan;
#endif
}

struct TokenLogit
{
    int id;
    float logit;
};

struct TokenProb
{
    int id;
    float prob;
};

struct MtpPattern
{
    std::string type;
    std::vector<int> tokens;
};

class LocatePassthroughLayer : public ncnn::Layer
{
public:
    LocatePassthroughLayer()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        if (bottom_blobs.empty())
            return -1;
        top_blobs[0] = bottom_blobs[0].clone(opt.blob_allocator);
        return top_blobs[0].empty() ? -100 : 0;
    }
};

class LocateExpressionLayer : public ncnn::Layer
{
public:
    LocateExpressionLayer()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>&, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(1, (size_t)4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        top_blob[0] = 0.f;
        return 0;
    }
};

DEFINE_LAYER_CREATOR(LocatePassthroughLayer)
DEFINE_LAYER_CREATOR(LocateExpressionLayer)

static void register_locateanything_custom_layers(ncnn::Net& net)
{
    net.register_custom_layer("pnnx.Expression", LocateExpressionLayer_layer_creator);
    net.register_custom_layer("aten::to", LocatePassthroughLayer_layer_creator);
    net.register_custom_layer("torch.view_as_complex", LocatePassthroughLayer_layer_creator);
    net.register_custom_layer("torch.view_as_real", LocatePassthroughLayer_layer_creator);
    net.register_custom_layer("Tensor.fill", LocatePassthroughLayer_layer_creator);
}

static std::string join_path(const std::string& dir, const std::string& name)
{
    if (dir.empty())
        return name;
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\')
        return dir + name;
    return dir + "/" + name;
}

static int file_exists(const std::string& path)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        return 0;
    fclose(fp);
    return 1;
}

static std::string model_file(const std::string& dir, const char* stem, const char* suffix, bool optimized)
{
    std::string name = stem;
    name += optimized ? ".opt.ncnn." : ".ncnn.";
    name += suffix;
    return join_path(dir, name);
}

static bool has_optimized_pair(const std::string& dir, const char* stem)
{
    return file_exists(model_file(dir, stem, "param", true)) && file_exists(model_file(dir, stem, "bin", true));
}

static int load_model(ncnn::Net& net, const std::string& dir, const char* stem, bool custom_layers, bool prefer_vulkan)
{
    if (custom_layers)
        register_locateanything_custom_layers(net);

    configure_net_runtime(net, prefer_vulkan);

    const bool optimized = has_optimized_pair(dir, stem);
    const std::string param = model_file(dir, stem, "param", optimized);
    const std::string bin = model_file(dir, stem, "bin", optimized);

    fprintf(stderr, "locate_anything: loading %s%s\n", param.c_str(),
#if NCNN_VULKAN
            net.opt.use_vulkan_compute ? " [vulkan]" : " [cpu]"
#else
            " [cpu]"
#endif
            );
    int ret = net.load_param(param.c_str());
    if (ret != 0)
        return ret;
    ret = net.load_model(bin.c_str());
    if (ret != 0)
        return ret;
    return 0;
}

static int load_model_cpu(ncnn::Net& net, const std::string& dir, const char* stem, bool custom_layers)
{
    return load_model(net, dir, stem, custom_layers, false);
}

static int run_blob(ncnn::Net& net, const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Extractor ex = net.create_extractor();
    int ret = ex.input("in0", in);
    if (ret != 0)
        return ret;
    return ex.extract("out0", out);
}

static float fp16_to_float(uint16_t h)
{
    const uint32_t s = (h >> 15) & 1;
    const uint32_t e = (h >> 10) & 31;
    const uint32_t f = h & 1023;
    if (e == 0)
    {
        if (f == 0) return s ? -0.f : 0.f;
        float v = std::ldexp((float)f, -24);
        return s ? -v : v;
    }
    if (e == 31)
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    float v = std::ldexp(1.f + (float)f / 1024.f, (int)e - 15);
    return s ? -v : v;
}

struct HalfTensor
{
    int rows;
    int cols;
    std::vector<uint16_t> data;
};

static std::string raw_weight_file(const std::string& model_dir, const std::string& name)
{
    std::string file = name;
    for (size_t i = 0; i < file.size(); i++)
        if (file[i] == '.')
            file.replace(i, 1, "__");
    file += ".f16.bin";
    return join_path(join_path(model_dir, "raw_runtime/weights_f16"), file);
}

static bool load_half_tensor(const std::string& model_dir, const std::string& name, int rows, int cols, HalfTensor& t)
{
    const std::string path = raw_weight_file(model_dir, name);
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in)
    {
        fprintf(stderr, "locate_anything: open raw weight failed %s\n", path.c_str());
        return false;
    }
    t.rows = rows;
    t.cols = cols;
    t.data.resize((size_t)rows * cols);
    in.read((char*)t.data.data(), (std::streamsize)t.data.size() * sizeof(uint16_t));
    if (!in)
    {
        fprintf(stderr, "locate_anything: short read %s\n", path.c_str());
        return false;
    }
    return true;
}

static bool load_half_tensor_path(const std::string& path, int rows, int cols, HalfTensor& t)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in)
        return false;
    t.rows = rows;
    t.cols = cols;
    t.data.resize((size_t)rows * cols);
    in.read((char*)t.data.data(), (std::streamsize)t.data.size() * sizeof(uint16_t));
    return !!in;
}

static bool load_embedding_row_from_table(const HalfTensor& table, int token, std::vector<float>& out)
{
    out.resize(qwen_hidden);
    for (int i = 0; i < qwen_hidden; i++)
        out[i] = fp16_to_float(table.data[(size_t)token * qwen_hidden + i]);
    return true;
}

static void linear_half(const std::vector<float>& x, const HalfTensor& w, const HalfTensor* bias, std::vector<float>& y)
{
    y.assign(w.rows, 0.f);
    for (int o = 0; o < w.rows; o++)
    {
        float sum = bias ? fp16_to_float(bias->data[o]) : 0.f;
        const uint16_t* row = w.data.data() + (size_t)o * w.cols;
        for (int i = 0; i < w.cols; i++)
            sum += fp16_to_float(row[i]) * x[i];
        y[o] = sum;
    }
}

static void rms_norm(std::vector<float>& x, const HalfTensor& weight)
{
    double ss = 0.0;
    for (size_t i = 0; i < x.size(); i++)
        ss += (double)x[i] * x[i];
    const float scale = 1.f / std::sqrt((float)(ss / x.size()) + qwen_rms_eps);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = x[i] * scale * fp16_to_float(weight.data[i]);
}

static float silu(float x)
{
    return x / (1.f + std::exp(-x));
}

static float gelu_tanh(float x)
{
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

static void layer_norm_affine(const float* src, float* dst, int n, const HalfTensor& weight, const HalfTensor& bias, float eps)
{
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += src[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++)
    {
        const double d = src[i] - mean;
        var += d * d;
    }
    const float scale = 1.f / std::sqrt((float)(var / n) + eps);
    for (int i = 0; i < n; i++)
        dst[i] = (src[i] - (float)mean) * scale * fp16_to_float(weight.data[i]) + fp16_to_float(bias.data[i]);
}

struct ImageResizeInfo
{
    int width;
    int height;
    int grid_w;
    int grid_h;
};

struct MoonViTLayerWeights
{
    HalfTensor norm0_w, norm0_b;
    HalfTensor norm1_w, norm1_b;
    HalfTensor wqkv_w, wqkv_b;
    HalfTensor wo_w, wo_b;
    HalfTensor fc0_w, fc0_b;
    HalfTensor fc1_w, fc1_b;
};

static bool load_moonvit_layer(const std::string& model_dir, int layer, MoonViTLayerWeights& w)
{
    char p[128];
    sprintf(p, "vision_model.encoder.blocks.%d.", layer);
    const std::string base = p;
    return load_half_tensor(model_dir, base + "norm0.weight", locate_vision_hidden_size, 1, w.norm0_w) &&
           load_half_tensor(model_dir, base + "norm0.bias", locate_vision_hidden_size, 1, w.norm0_b) &&
           load_half_tensor(model_dir, base + "norm1.weight", locate_vision_hidden_size, 1, w.norm1_w) &&
           load_half_tensor(model_dir, base + "norm1.bias", locate_vision_hidden_size, 1, w.norm1_b) &&
           load_half_tensor(model_dir, base + "wqkv.weight", locate_vision_hidden_size * 3, locate_vision_hidden_size, w.wqkv_w) &&
           load_half_tensor(model_dir, base + "wqkv.bias", locate_vision_hidden_size * 3, 1, w.wqkv_b) &&
           load_half_tensor(model_dir, base + "wo.weight", locate_vision_hidden_size, locate_vision_hidden_size, w.wo_w) &&
           load_half_tensor(model_dir, base + "wo.bias", locate_vision_hidden_size, 1, w.wo_b) &&
           load_half_tensor(model_dir, base + "mlp.fc0.weight", moonvit_mlp_hidden, locate_vision_hidden_size, w.fc0_w) &&
           load_half_tensor(model_dir, base + "mlp.fc0.bias", moonvit_mlp_hidden, 1, w.fc0_b) &&
           load_half_tensor(model_dir, base + "mlp.fc1.weight", locate_vision_hidden_size, moonvit_mlp_hidden, w.fc1_w) &&
           load_half_tensor(model_dir, base + "mlp.fc1.bias", locate_vision_hidden_size, 1, w.fc1_b);
}

static int round_down_multiple(int x, int m)
{
    return std::max(m, (x / m) * m);
}

static ImageResizeInfo choose_image_resize(int w, int h)
{
    const int unit = locate_patch_size * locate_merge_w;
    int patch_limit = locate_default_cpu_patch_limit;
    const char* env_limit = getenv("LOCATEANYTHING_MAX_PATCHES");
    if (env_limit && env_limit[0])
    {
        patch_limit = atoi(env_limit);
        if (patch_limit <= 0)
            patch_limit = locate_default_cpu_patch_limit;
        patch_limit = std::min(patch_limit, locate_image_token_limit);
    }

    float scale = 1.f;
    ImageResizeInfo info;
    for (;;)
    {
        info.width = round_down_multiple((int)std::floor(w * scale + 0.5f), unit);
        info.height = round_down_multiple((int)std::floor(h * scale + 0.5f), unit);
        info.grid_w = info.width / locate_patch_size;
        info.grid_h = info.height / locate_patch_size;
        if (info.grid_w * info.grid_h <= patch_limit)
            return info;
        scale *= 0.9f;
    }
}

static ncnn::Mat make_dynamic_image_input(const cv::Mat& bgr, ImageResizeInfo& info)
{
    info = choose_image_resize(bgr.cols, bgr.rows);
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, info.width, info.height);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    resized.substract_mean_normalize(mean_vals, norm_vals);
    return resized;
}

static void patch_conv_output_to_tokens(const ncnn::Mat& conv_out, int grid_h, int grid_w, std::vector<float>& tokens)
{
    tokens.assign((size_t)grid_h * grid_w * locate_vision_hidden_size, 0.f);
    for (int c = 0; c < locate_vision_hidden_size; c++)
    {
        const ncnn::Mat ch = conv_out.channel(c);
        for (int y = 0; y < grid_h; y++)
        {
            const float* row = ch.row(y);
            for (int x = 0; x < grid_w; x++)
                tokens[((size_t)y * grid_w + x) * locate_vision_hidden_size + c] = row[x];
        }
    }
}

static void add_interpolated_pos_embed(std::vector<float>& tokens, int grid_h, int grid_w, const HalfTensor& pos)
{
    // Match torch bicubic interpolate with align_corners=False for the learned
    // 64x64 MoonViT absolute position embedding.
    const float cubic_a = -0.75f;
    const auto cubic = [cubic_a](float x) -> float {
        x = std::fabs(x);
        if (x <= 1.f)
            return (cubic_a + 2.f) * x * x * x - (cubic_a + 3.f) * x * x + 1.f;
        if (x < 2.f)
            return cubic_a * x * x * x - 5.f * cubic_a * x * x + 8.f * cubic_a * x - 4.f * cubic_a;
        return 0.f;
    };

    for (int y = 0; y < grid_h; y++)
    {
        const float fy = ((float)y + 0.5f) * moonvit_pos_grid / grid_h - 0.5f;
        const int ybase = (int)std::floor(fy);
        for (int x = 0; x < grid_w; x++)
        {
            const float fx = ((float)x + 0.5f) * moonvit_pos_grid / grid_w - 0.5f;
            const int xbase = (int)std::floor(fx);
            float* dst = tokens.data() + ((size_t)y * grid_w + x) * locate_vision_hidden_size;
            for (int c = 0; c < locate_vision_hidden_size; c++)
            {
                float sum = 0.f;
                for (int ky = -1; ky <= 2; ky++)
                {
                    const int sy = std::min(std::max(ybase + ky, 0), moonvit_pos_grid - 1);
                    const float wy = cubic(fy - (ybase + ky));
                    for (int kx = -1; kx <= 2; kx++)
                    {
                        const int sx = std::min(std::max(xbase + kx, 0), moonvit_pos_grid - 1);
                        const float wx = cubic(fx - (xbase + kx));
                        const uint16_t* p = pos.data.data() + ((size_t)sy * moonvit_pos_grid + sx) * locate_vision_hidden_size;
                        sum += wy * wx * fp16_to_float(p[c]);
                    }
                }
                dst[c] += sum;
            }
        }
    }
}

static void apply_moonvit_rope_one(float* v, int grid_y, int grid_x)
{
    for (int i = 0; i + 1 < moonvit_head_dim; i += 2)
    {
        const int pair_index = i / 2;
        const bool y_axis = (pair_index % 2) == 0;
        const int axis_pos = y_axis ? grid_y : grid_x;
        const int axis_i = (pair_index / 2) * 2;
        const float inv_freq = std::pow(10000.f, -(float)axis_i / (moonvit_head_dim / 2));
        const float angle = axis_pos * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float a = v[i];
        const float b = v[i + 1];
        v[i] = a * c - b * s;
        v[i + 1] = a * s + b * c;
    }
}

static void moonvit_attention(const std::vector<float>& qkv, int seq, int grid_h, int grid_w, std::vector<float>& out)
{
    out.assign((size_t)seq * locate_vision_hidden_size, 0.f);
    std::vector<float> scores(seq);
    const float inv_sqrt = 1.f / std::sqrt((float)moonvit_head_dim);
    for (int h = 0; h < moonvit_heads; h++)
    {
        for (int t = 0; t < seq; t++)
        {
            std::vector<float> q(moonvit_head_dim);
            const float* qsrc = qkv.data() + ((size_t)t * 3 * locate_vision_hidden_size + h * moonvit_head_dim);
            memcpy(q.data(), qsrc, moonvit_head_dim * sizeof(float));
            apply_moonvit_rope_one(q.data(), t / grid_w, t % grid_w);

            float max_score = -INFINITY;
            for (int sidx = 0; sidx < seq; sidx++)
            {
                std::vector<float> k(moonvit_head_dim);
                const float* ksrc = qkv.data() + ((size_t)sidx * 3 * locate_vision_hidden_size + locate_vision_hidden_size + h * moonvit_head_dim);
                memcpy(k.data(), ksrc, moonvit_head_dim * sizeof(float));
                apply_moonvit_rope_one(k.data(), sidx / grid_w, sidx % grid_w);
                float dot = 0.f;
                for (int d = 0; d < moonvit_head_dim; d++)
                    dot += q[d] * k[d];
                scores[sidx] = dot * inv_sqrt;
                max_score = std::max(max_score, scores[sidx]);
            }
            float denom = 0.f;
            for (int sidx = 0; sidx < seq; sidx++)
            {
                scores[sidx] = std::exp(scores[sidx] - max_score);
                denom += scores[sidx];
            }
            float* dst = out.data() + (size_t)t * locate_vision_hidden_size + h * moonvit_head_dim;
            for (int sidx = 0; sidx < seq; sidx++)
            {
                const float a = scores[sidx] / denom;
                const float* v = qkv.data() + ((size_t)sidx * 3 * locate_vision_hidden_size + 2 * locate_vision_hidden_size + h * moonvit_head_dim);
                for (int d = 0; d < moonvit_head_dim; d++)
                    dst[d] += a * v[d];
            }
        }
    }
}

static void moonvit_layer_forward(std::vector<float>& tokens, int grid_h, int grid_w, const MoonViTLayerWeights& w)
{
    const int seq = grid_h * grid_w;
    std::vector<float> normed((size_t)seq * locate_vision_hidden_size);
    for (int i = 0; i < seq; i++)
        layer_norm_affine(tokens.data() + (size_t)i * locate_vision_hidden_size, normed.data() + (size_t)i * locate_vision_hidden_size,
                          locate_vision_hidden_size, w.norm0_w, w.norm0_b, 1e-5f);

    std::vector<float> qkv((size_t)seq * locate_vision_hidden_size * 3);
    std::vector<float> tmp;
    for (int i = 0; i < seq; i++)
    {
        std::vector<float> x(normed.begin() + (size_t)i * locate_vision_hidden_size, normed.begin() + ((size_t)i + 1) * locate_vision_hidden_size);
        linear_half(x, w.wqkv_w, &w.wqkv_b, tmp);
        memcpy(qkv.data() + (size_t)i * locate_vision_hidden_size * 3, tmp.data(), tmp.size() * sizeof(float));
    }

    std::vector<float> attn;
    moonvit_attention(qkv, seq, grid_h, grid_w, attn);
    for (int i = 0; i < seq; i++)
    {
        std::vector<float> x(attn.begin() + (size_t)i * locate_vision_hidden_size, attn.begin() + ((size_t)i + 1) * locate_vision_hidden_size);
        linear_half(x, w.wo_w, &w.wo_b, tmp);
        float* dst = tokens.data() + (size_t)i * locate_vision_hidden_size;
        for (int c = 0; c < locate_vision_hidden_size; c++)
            dst[c] += tmp[c];
    }

    for (int i = 0; i < seq; i++)
        layer_norm_affine(tokens.data() + (size_t)i * locate_vision_hidden_size, normed.data() + (size_t)i * locate_vision_hidden_size,
                          locate_vision_hidden_size, w.norm1_w, w.norm1_b, 1e-5f);
    for (int i = 0; i < seq; i++)
    {
        std::vector<float> x(normed.begin() + (size_t)i * locate_vision_hidden_size, normed.begin() + ((size_t)i + 1) * locate_vision_hidden_size);
        std::vector<float> hidden;
        linear_half(x, w.fc0_w, &w.fc0_b, hidden);
        for (int j = 0; j < moonvit_mlp_hidden; j++)
            hidden[j] = gelu_tanh(hidden[j]);
        linear_half(hidden, w.fc1_w, &w.fc1_b, tmp);
        float* dst = tokens.data() + (size_t)i * locate_vision_hidden_size;
        for (int c = 0; c < locate_vision_hidden_size; c++)
            dst[c] += tmp[c];
    }
}

static int project_vision_token(ncnn::Net& projector, const float* merged, std::vector<float>& out)
{
    ncnn::Mat in(locate_vision_hidden_size * locate_merge_h * locate_merge_w);
    if (in.empty())
        return -1;
    memcpy((float*)in, merged, (size_t)locate_vision_hidden_size * locate_merge_h * locate_merge_w * sizeof(float));
    ncnn::Mat projected;
    int ret = run_blob(projector, in, projected);
    if (ret != 0 || projected.empty())
        return ret ? ret : -1;
    out.resize(qwen_hidden);
    memcpy(out.data(), (const float*)projected, qwen_hidden * sizeof(float));
    return 0;
}

static int project_vision_token_with_fallback(ncnn::Net& projector, bool& projector_vulkan, const std::string& model_dir, const float* merged, std::vector<float>& out)
{
    int ret = project_vision_token(projector, merged, out);
    if (ret == 0)
        return 0;

    if (!projector_vulkan)
        return ret;

    fprintf(stderr, "locate_anything: projector Vulkan failed ret=%d, reload CPU projector\n", ret);
    projector.clear();
    if (load_model_cpu(projector, model_dir, "locateanything_projector_token", false) != 0)
        return ret;
    projector_vulkan = false;
    return project_vision_token(projector, merged, out);
}

static int run_image_embedding_dynamic(const std::string& model_dir, const cv::Mat& bgr, std::vector<std::vector<float> >& image_embeds, ImageResizeInfo& info)
{
    ncnn::Net patch_conv;
    ncnn::Net projector;
    bool patch_conv_vulkan = locate_vulkan_enabled();
    bool projector_vulkan = locate_vulkan_enabled();
    if (load_model(patch_conv, model_dir, "moonvit_patch_conv", false, patch_conv_vulkan) != 0) return -1;
    if (load_model(projector, model_dir, "locateanything_projector_token", false, projector_vulkan) != 0) return -1;

    ncnn::Mat image = make_dynamic_image_input(bgr, info);
    ncnn::Mat conv_out;
    int ret = run_blob(patch_conv, image, conv_out);
    if ((ret != 0 || conv_out.empty()) && patch_conv_vulkan)
    {
        fprintf(stderr, "locate_anything: patch_conv Vulkan failed ret=%d empty=%d, reload CPU patch_conv\n", ret, conv_out.empty() ? 1 : 0);
        patch_conv.clear();
        if (load_model_cpu(patch_conv, model_dir, "moonvit_patch_conv", false) != 0)
            return ret ? ret : -1;
        patch_conv_vulkan = false;
        ret = run_blob(patch_conv, image, conv_out);
    }
    if (ret != 0 || conv_out.empty())
        return ret ? ret : -1;

    std::vector<float> tokens;
    patch_conv_output_to_tokens(conv_out, info.grid_h, info.grid_w, tokens);

    HalfTensor pos;
    if (!load_half_tensor(model_dir, "vision_model.patch_embed.pos_emb.weight", moonvit_pos_grid * moonvit_pos_grid, locate_vision_hidden_size, pos))
        return -1;
    add_interpolated_pos_embed(tokens, info.grid_h, info.grid_w, pos);

    for (int i = 0; i < moonvit_layers; i++)
    {
        MoonViTLayerWeights w;
        if (!load_moonvit_layer(model_dir, i, w))
            return -1;
        moonvit_layer_forward(tokens, info.grid_h, info.grid_w, w);
        fprintf(stderr, "locate_anything: moonvit layer %d/%d done\n", i + 1, moonvit_layers);
    }

    HalfTensor final_w, final_b;
    if (!load_half_tensor(model_dir, "vision_model.encoder.final_layernorm.weight", locate_vision_hidden_size, 1, final_w) ||
        !load_half_tensor(model_dir, "vision_model.encoder.final_layernorm.bias", locate_vision_hidden_size, 1, final_b))
        return -1;

    const int merged_h = info.grid_h / locate_merge_h;
    const int merged_w = info.grid_w / locate_merge_w;
    image_embeds.clear();
    image_embeds.reserve((size_t)merged_h * merged_w);
    std::vector<float> merged(locate_vision_hidden_size * locate_merge_h * locate_merge_w);
    std::vector<float> normed(locate_vision_hidden_size);
    for (int gy = 0; gy < merged_h; gy++)
    {
        for (int gx = 0; gx < merged_w; gx++)
        {
            float* mp = merged.data();
            for (int dy = 0; dy < locate_merge_h; dy++)
            {
                for (int dx = 0; dx < locate_merge_w; dx++)
                {
                    const int y = gy * locate_merge_h + dy;
                    const int x = gx * locate_merge_w + dx;
                    layer_norm_affine(tokens.data() + ((size_t)y * info.grid_w + x) * locate_vision_hidden_size,
                                      normed.data(), locate_vision_hidden_size, final_w, final_b, 1e-5f);
                    memcpy(mp, normed.data(), locate_vision_hidden_size * sizeof(float));
                    mp += locate_vision_hidden_size;
                }
            }
            std::vector<float> projected;
            if (project_vision_token_with_fallback(projector, projector_vulkan, model_dir, merged.data(), projected) != 0)
                return -1;
            image_embeds.push_back(projected);
        }
    }

    fprintf(stderr, "locate_anything: dynamic image %dx%d -> grid %dx%d -> %d image tokens\n",
            info.width, info.height, info.grid_h, info.grid_w, (int)image_embeds.size());
    return 0;
}

struct KVCacheLayer
{
    std::vector<float> key;
    std::vector<float> value;
    int seq;
    KVCacheLayer() : seq(0) {}
};

struct QwenLayerWeights
{
    HalfTensor input_norm, post_norm;
    HalfTensor q_w, q_b, k_w, k_b, v_w, v_b, o_w;
    HalfTensor gate_w, up_w, down_w;
};

static bool load_qwen_layer(const std::string& model_dir, int layer, QwenLayerWeights& w)
{
    char p[128];
    sprintf(p, "language_model.model.layers.%d.", layer);
    const std::string base = p;
    return load_half_tensor(model_dir, base + "input_layernorm.weight", qwen_hidden, 1, w.input_norm) &&
           load_half_tensor(model_dir, base + "post_attention_layernorm.weight", qwen_hidden, 1, w.post_norm) &&
           load_half_tensor(model_dir, base + "self_attn.q_proj.weight", qwen_hidden, qwen_hidden, w.q_w) &&
           load_half_tensor(model_dir, base + "self_attn.q_proj.bias", qwen_hidden, 1, w.q_b) &&
           load_half_tensor(model_dir, base + "self_attn.k_proj.weight", qwen_kv_heads * qwen_head_dim, qwen_hidden, w.k_w) &&
           load_half_tensor(model_dir, base + "self_attn.k_proj.bias", qwen_kv_heads * qwen_head_dim, 1, w.k_b) &&
           load_half_tensor(model_dir, base + "self_attn.v_proj.weight", qwen_kv_heads * qwen_head_dim, qwen_hidden, w.v_w) &&
           load_half_tensor(model_dir, base + "self_attn.v_proj.bias", qwen_kv_heads * qwen_head_dim, 1, w.v_b) &&
           load_half_tensor(model_dir, base + "self_attn.o_proj.weight", qwen_hidden, qwen_hidden, w.o_w) &&
           load_half_tensor(model_dir, base + "mlp.gate_proj.weight", qwen_intermediate, qwen_hidden, w.gate_w) &&
           load_half_tensor(model_dir, base + "mlp.up_proj.weight", qwen_intermediate, qwen_hidden, w.up_w) &&
           load_half_tensor(model_dir, base + "mlp.down_proj.weight", qwen_hidden, qwen_intermediate, w.down_w);
}

static void apply_qwen_rope(std::vector<float>& q, std::vector<float>& k, int position)
{
    const int half = qwen_head_dim / 2;
    for (int i = 0; i < half; i++)
    {
        const float inv_freq = std::pow(qwen_rope_theta, -(float)(i * 2) / qwen_head_dim);
        const float angle = position * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        for (int h = 0; h < qwen_heads; h++)
        {
            float& x0 = q[(size_t)h * qwen_head_dim + i];
            float& x1 = q[(size_t)h * qwen_head_dim + half + i];
            const float a = x0;
            const float b = x1;
            x0 = a * c - b * s;
            x1 = a * s + b * c;
        }
        for (int h = 0; h < qwen_kv_heads; h++)
        {
            float& x0 = k[(size_t)h * qwen_head_dim + i];
            float& x1 = k[(size_t)h * qwen_head_dim + half + i];
            const float a = x0;
            const float b = x1;
            x0 = a * c - b * s;
            x1 = a * s + b * c;
        }
    }
}

static void attention_decode_one(const std::vector<float>& q, const std::vector<float>& k_new, const std::vector<float>& v_new, KVCacheLayer& cache, std::vector<float>& out)
{
    const int old_seq = cache.seq;
    const int new_seq = old_seq + 1;
    cache.key.resize((size_t)new_seq * qwen_kv_heads * qwen_head_dim);
    cache.value.resize((size_t)new_seq * qwen_kv_heads * qwen_head_dim);
    memcpy(cache.key.data() + (size_t)old_seq * qwen_kv_heads * qwen_head_dim, k_new.data(), k_new.size() * sizeof(float));
    memcpy(cache.value.data() + (size_t)old_seq * qwen_kv_heads * qwen_head_dim, v_new.data(), v_new.size() * sizeof(float));
    cache.seq = new_seq;

    out.assign(qwen_hidden, 0.f);
    std::vector<float> scores(new_seq);
    const int group = qwen_heads / qwen_kv_heads;
    const float inv_sqrt = 1.f / std::sqrt((float)qwen_head_dim);
    for (int h = 0; h < qwen_heads; h++)
    {
        const int kvh = h / group;
        float max_score = -INFINITY;
        for (int t = 0; t < new_seq; t++)
        {
            float dot = 0.f;
            const float* qp = q.data() + (size_t)h * qwen_head_dim;
            const float* kp = cache.key.data() + ((size_t)t * qwen_kv_heads + kvh) * qwen_head_dim;
            for (int d = 0; d < qwen_head_dim; d++)
                dot += qp[d] * kp[d];
            scores[t] = dot * inv_sqrt;
            max_score = std::max(max_score, scores[t]);
        }
        float denom = 0.f;
        for (int t = 0; t < new_seq; t++)
        {
            scores[t] = std::exp(scores[t] - max_score);
            denom += scores[t];
        }
        float* op = out.data() + (size_t)h * qwen_head_dim;
        for (int t = 0; t < new_seq; t++)
        {
            const float a = scores[t] / denom;
            const float* vp = cache.value.data() + ((size_t)t * qwen_kv_heads + kvh) * qwen_head_dim;
            for (int d = 0; d < qwen_head_dim; d++)
                op[d] += a * vp[d];
        }
    }
}

static void attention_decode_block(const std::vector<float>& q, const std::vector<float>& k_new, const std::vector<float>& v_new, int q_len, KVCacheLayer& cache, std::vector<float>& out)
{
    const int old_seq = cache.seq;
    const int new_seq = old_seq + q_len;
    cache.key.resize((size_t)new_seq * qwen_kv_heads * qwen_head_dim);
    cache.value.resize((size_t)new_seq * qwen_kv_heads * qwen_head_dim);
    memcpy(cache.key.data() + (size_t)old_seq * qwen_kv_heads * qwen_head_dim, k_new.data(), k_new.size() * sizeof(float));
    memcpy(cache.value.data() + (size_t)old_seq * qwen_kv_heads * qwen_head_dim, v_new.data(), v_new.size() * sizeof(float));
    cache.seq = new_seq;

    out.assign((size_t)q_len * qwen_hidden, 0.f);
    std::vector<float> scores(new_seq);
    const int group = qwen_heads / qwen_kv_heads;
    const float inv_sqrt = 1.f / std::sqrt((float)qwen_head_dim);
    for (int qi = 0; qi < q_len; qi++)
    {
        const int allowed_seq = old_seq + qi + 1;
        for (int h = 0; h < qwen_heads; h++)
        {
            const int kvh = h / group;
            float max_score = -INFINITY;
            for (int t = 0; t < allowed_seq; t++)
            {
                float dot = 0.f;
                const float* qp = q.data() + ((size_t)qi * qwen_heads + h) * qwen_head_dim;
                const float* kp = cache.key.data() + ((size_t)t * qwen_kv_heads + kvh) * qwen_head_dim;
                for (int d = 0; d < qwen_head_dim; d++)
                    dot += qp[d] * kp[d];
                scores[t] = dot * inv_sqrt;
                max_score = std::max(max_score, scores[t]);
            }
            float denom = 0.f;
            for (int t = 0; t < allowed_seq; t++)
            {
                scores[t] = std::exp(scores[t] - max_score);
                denom += scores[t];
            }
            float* op = out.data() + ((size_t)qi * qwen_heads + h) * qwen_head_dim;
            for (int t = 0; t < allowed_seq; t++)
            {
                const float a = scores[t] / denom;
                const float* vp = cache.value.data() + ((size_t)t * qwen_kv_heads + kvh) * qwen_head_dim;
                for (int d = 0; d < qwen_head_dim; d++)
                    op[d] += a * vp[d];
            }
        }
    }
}

static void qwen_layer_forward_one(const std::vector<float>& input, const QwenLayerWeights& w, KVCacheLayer& cache, int position, std::vector<float>& output)
{
    std::vector<float> x = input;
    rms_norm(x, w.input_norm);

    std::vector<float> q, k, v;
    linear_half(x, w.q_w, &w.q_b, q);
    linear_half(x, w.k_w, &w.k_b, k);
    linear_half(x, w.v_w, &w.v_b, v);
    apply_qwen_rope(q, k, position);

    std::vector<float> attn, attn_proj;
    attention_decode_one(q, k, v, cache, attn);
    linear_half(attn, w.o_w, 0, attn_proj);

    output.resize(qwen_hidden);
    for (int i = 0; i < qwen_hidden; i++)
        output[i] = input[i] + attn_proj[i];

    std::vector<float> z = output;
    rms_norm(z, w.post_norm);
    std::vector<float> gate, up, mlp;
    linear_half(z, w.gate_w, 0, gate);
    linear_half(z, w.up_w, 0, up);
    for (int i = 0; i < qwen_intermediate; i++)
        gate[i] = silu(gate[i]) * up[i];
    linear_half(gate, w.down_w, 0, mlp);
    for (int i = 0; i < qwen_hidden; i++)
        output[i] += mlp[i];
}

static void qwen_layer_forward_block(const std::vector<float>& input, int q_len, const QwenLayerWeights& w, KVCacheLayer& cache, const std::vector<int>& positions, std::vector<float>& output)
{
    std::vector<float> normed((size_t)q_len * qwen_hidden);
    std::vector<float> q((size_t)q_len * qwen_heads * qwen_head_dim);
    std::vector<float> k((size_t)q_len * qwen_kv_heads * qwen_head_dim);
    std::vector<float> v((size_t)q_len * qwen_kv_heads * qwen_head_dim);

    for (int i = 0; i < q_len; i++)
    {
        std::vector<float> x(input.begin() + (size_t)i * qwen_hidden, input.begin() + ((size_t)i + 1) * qwen_hidden);
        rms_norm(x, w.input_norm);
        memcpy(normed.data() + (size_t)i * qwen_hidden, x.data(), qwen_hidden * sizeof(float));

        std::vector<float> qi, ki, vi;
        linear_half(x, w.q_w, &w.q_b, qi);
        linear_half(x, w.k_w, &w.k_b, ki);
        linear_half(x, w.v_w, &w.v_b, vi);
        apply_qwen_rope(qi, ki, positions[i]);
        memcpy(q.data() + (size_t)i * qwen_heads * qwen_head_dim, qi.data(), qi.size() * sizeof(float));
        memcpy(k.data() + (size_t)i * qwen_kv_heads * qwen_head_dim, ki.data(), ki.size() * sizeof(float));
        memcpy(v.data() + (size_t)i * qwen_kv_heads * qwen_head_dim, vi.data(), vi.size() * sizeof(float));
    }

    std::vector<float> attn;
    attention_decode_block(q, k, v, q_len, cache, attn);

    output.resize((size_t)q_len * qwen_hidden);
    for (int i = 0; i < q_len; i++)
    {
        std::vector<float> attn_i(attn.begin() + (size_t)i * qwen_hidden, attn.begin() + ((size_t)i + 1) * qwen_hidden);
        std::vector<float> attn_proj;
        linear_half(attn_i, w.o_w, 0, attn_proj);
        float* dst = output.data() + (size_t)i * qwen_hidden;
        const float* src = input.data() + (size_t)i * qwen_hidden;
        for (int j = 0; j < qwen_hidden; j++)
            dst[j] = src[j] + attn_proj[j];

        std::vector<float> z(dst, dst + qwen_hidden);
        rms_norm(z, w.post_norm);
        std::vector<float> gate, up, mlp;
        linear_half(z, w.gate_w, 0, gate);
        linear_half(z, w.up_w, 0, up);
        for (int j = 0; j < qwen_intermediate; j++)
            gate[j] = silu(gate[j]) * up[j];
        linear_half(gate, w.down_w, 0, mlp);
        for (int j = 0; j < qwen_hidden; j++)
            dst[j] += mlp[j];
    }
}

class QwenTokenizer
{
public:
    bool load(const char* vocab_path, const char* merges_path)
    {
        generate_byte_decoder();
        std::string json;
        if (!read_file(vocab_path, json) || !parse_vocab_json(json))
            return false;
        FILE* fp = fopen(merges_path, "rb");
        if (!fp)
            return false;
        char line[4096];
        int rank = 0;
        while (fgets(line, sizeof(line), fp))
        {
            std::string s = trim_newline(line);
            if (s.empty() || s[0] == '#')
                continue;
            size_t sp = s.find(' ');
            if (sp != std::string::npos)
                bpe_ranks[std::make_pair(s.substr(0, sp), s.substr(sp + 1))] = rank++;
        }
        fclose(fp);
        return true;
    }

    std::vector<int> encode(const std::string& text) const
    {
        std::vector<int> ids;
        std::vector<std::string> pieces = split_text(text);
        for (size_t i = 0; i < pieces.size(); i++)
            encode_piece(pieces[i], ids);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const
    {
        std::string bytes_text;
        for (int id : ids)
        {
            if (id >= (int)id_to_token.size())
                continue;
            if (id >= token_endoftext)
                continue;
            bytes_text += id_to_token[id];
        }
        std::vector<uint32_t> cps = utf8_to_codepoints(bytes_text);
        std::string out;
        for (size_t i = 0; i < cps.size(); i++)
        {
            std::map<uint32_t, unsigned char>::const_iterator it = byte_decoder.find(cps[i]);
            if (it != byte_decoder.end())
                out.push_back((char)it->second);
        }
        return out;
    }

private:
    void encode_piece(const std::string& text, std::vector<int>& ids) const
    {
        std::string bytes;
        for (size_t i = 0; i < text.size(); i++)
            append_utf8(bytes, byte_encoder[(unsigned char)text[i]]);
        std::vector<std::string> word;
        for (size_t i = 0; i < bytes.size();)
        {
            int len = utf8_char_len((unsigned char)bytes[i]);
            word.push_back(bytes.substr(i, len));
            i += len;
        }
        for (;;)
        {
            int best_rank = 2147483647;
            int best_index = -1;
            for (int i = 0; i + 1 < (int)word.size(); i++)
            {
                std::map<std::pair<std::string, std::string>, int>::const_iterator it = bpe_ranks.find(std::make_pair(word[i], word[i + 1]));
                if (it != bpe_ranks.end() && it->second < best_rank)
                {
                    best_rank = it->second;
                    best_index = i;
                }
            }
            if (best_index == -1)
                break;
            word[best_index] += word[best_index + 1];
            word.erase(word.begin() + best_index + 1);
        }
        for (size_t i = 0; i < word.size(); i++)
        {
            std::map<std::string, int>::const_iterator it = vocab.find(word[i]);
            if (it != vocab.end())
                ids.push_back(it->second);
        }
    }

    static bool read_file(const char* path, std::string& content)
    {
        FILE* fp = fopen(path, "rb");
        if (!fp)
            return false;
        fseek(fp, 0, SEEK_END);
        long len = ftell(fp);
        rewind(fp);
        content.resize(len);
        fread(&content[0], 1, len, fp);
        fclose(fp);
        return true;
    }

    static std::string trim_newline(const char* s)
    {
        std::string r = s;
        while (!r.empty() && (r.back() == '\n' || r.back() == '\r'))
            r.pop_back();
        return r;
    }

    static int utf8_char_len(unsigned char c)
    {
        if (c < 0x80) return 1;
        if ((c & 0xe0) == 0xc0) return 2;
        if ((c & 0xf0) == 0xe0) return 3;
        if ((c & 0xf8) == 0xf0) return 4;
        return 1;
    }

    static void append_utf8(std::string& s, uint32_t cp)
    {
        if (cp < 0x80) s.push_back((char)cp);
        else if (cp < 0x800)
        {
            s.push_back((char)(0xc0 | (cp >> 6)));
            s.push_back((char)(0x80 | (cp & 0x3f)));
        }
        else
        {
            s.push_back((char)(0xe0 | (cp >> 12)));
            s.push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
            s.push_back((char)(0x80 | (cp & 0x3f)));
        }
    }

    static bool is_ascii_word(unsigned char c)
    {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    }

    static std::vector<std::string> split_text(const std::string& text)
    {
        std::vector<std::string> pieces;
        for (size_t i = 0; i < text.size();)
        {
            size_t start = i;
            unsigned char c = (unsigned char)text[i];
            if (c == ' ' || c == '\t')
            {
                while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) i++;
                if (i < text.size() && is_ascii_word((unsigned char)text[i]))
                    while (i < text.size() && is_ascii_word((unsigned char)text[i])) i++;
                pieces.push_back(text.substr(start, i - start));
                continue;
            }
            if (c == '\r' || c == '\n')
            {
                while (i < text.size() && (text[i] == '\r' || text[i] == '\n')) i++;
                pieces.push_back(text.substr(start, i - start));
                continue;
            }
            if (is_ascii_word(c))
            {
                while (i < text.size() && is_ascii_word((unsigned char)text[i])) i++;
                pieces.push_back(text.substr(start, i - start));
                continue;
            }
            int len = utf8_char_len(c);
            i += std::max(1, len);
            pieces.push_back(text.substr(start, i - start));
        }
        return pieces;
    }

    static std::vector<uint32_t> utf8_to_codepoints(const std::string& s)
    {
        std::vector<uint32_t> cps;
        for (size_t i = 0; i < s.size();)
        {
            unsigned char c = (unsigned char)s[i];
            int len = utf8_char_len(c);
            uint32_t cp = c;
            if (len == 2 && i + 1 < s.size()) cp = ((s[i] & 0x1f) << 6) | (s[i + 1] & 0x3f);
            else if (len == 3 && i + 2 < s.size()) cp = ((s[i] & 0x0f) << 12) | ((s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f);
            cps.push_back(cp);
            i += len;
        }
        return cps;
    }

    void generate_byte_decoder()
    {
        std::vector<int> bs;
        for (int i = '!'; i <= '~'; i++) bs.push_back(i);
        for (int i = 161; i <= 172; i++) bs.push_back(i);
        for (int i = 174; i <= 255; i++) bs.push_back(i);
        std::set<int> used(bs.begin(), bs.end());
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++)
            if (!used.count(b))
            {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        for (size_t i = 0; i < bs.size(); i++)
        {
            byte_encoder[bs[i]] = cs[i];
            byte_decoder[cs[i]] = (unsigned char)bs[i];
        }
    }

    bool parse_vocab_json(const std::string& json)
    {
        size_t i = 0;
        while (i < json.size())
        {
            if (json[i] != '"') { i++; continue; }
            i++;
            std::string key;
            while (i < json.size() && json[i] != '"')
            {
                if (json[i] == '\\' && i + 1 < json.size()) i++;
                key.push_back(json[i++]);
            }
            i++;
            while (i < json.size() && json[i] != ':') i++;
            i++;
            int id = atoi(json.c_str() + i);
            vocab[key] = id;
            if (id >= (int)id_to_token.size()) id_to_token.resize(id + 1);
            id_to_token[id] = key;
        }
        return !vocab.empty();
    }

    uint32_t byte_encoder[256];
    std::map<uint32_t, unsigned char> byte_decoder;
    std::map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
};

class LocateQwen
{
public:
    bool load(const std::string& dir)
    {
        model_dir = dir;
        if (!load_half_tensor(model_dir, "language_model.model.norm.weight", qwen_hidden, 1, final_norm))
            return false;
        if (!load_half_tensor_path(raw_weight_file(model_dir, "language_model.model.embed_tokens.weight"), qwen_vocab, qwen_hidden, embedding_table))
            return false;
        if (!load_half_tensor_path(raw_weight_file(model_dir, "language_model.lm_head.weight"), qwen_vocab, qwen_hidden, lm_head_table))
            return false;
        lm_head_runtime_available = false;
        const bool request_lm_head_vulkan = locate_env_flag("LOCATEANYTHING_VULKAN_LM_HEAD", false);
        lm_head_runtime_vulkan = request_lm_head_vulkan && locate_vulkan_enabled();
        lm_head_batch_runtime_enabled = locate_env_flag("LOCATEANYTHING_VULKAN_LM_HEAD_BATCH", false);
        if (lm_head_runtime_vulkan && load_model(lm_head_runtime, model_dir, "lm_head_token", false, true) == 0)
        {
            lm_head_runtime_available = true;
            fprintf(stderr, "locate_anything: lm_head runtime enabled via ncnn Vulkan\n");
            if (!lm_head_batch_runtime_enabled)
                fprintf(stderr, "locate_anything: lm_head batch runtime disabled by default; set LOCATEANYTHING_VULKAN_LM_HEAD_BATCH=1 to test MTP batch logits\n");
        }
        else
        {
            if (request_lm_head_vulkan)
                fprintf(stderr, "locate_anything: lm_head runtime load failed, fallback to raw CPU lm_head\n");
            else
                fprintf(stderr, "locate_anything: lm_head Vulkan disabled by default; set LOCATEANYTHING_VULKAN_LM_HEAD=1 to test it\n");
            lm_head_runtime.clear();
            lm_head_runtime_available = false;
            lm_head_runtime_vulkan = false;
        }
        if (!tokenizer.load(join_path(join_path(model_dir, "runtime_assets"), "vocab.json").c_str(),
                            join_path(join_path(model_dir, "runtime_assets"), "merges.txt").c_str()))
            return false;
        cache.resize(qwen_layers);
        layer_weights.resize(qwen_layers);
        layer_loaded.resize(qwen_layers, false);
        return true;
    }

    std::vector<int> build_prompt(const std::string& query, int image_token_count) const
    {
        std::vector<int> prompt;
        append_chat_text(prompt, "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
        prompt.push_back(token_im_end);
        append_tokens(prompt, tokenizer.encode("\n"));
        prompt.push_back(token_im_start);
        append_tokens(prompt, tokenizer.encode("user\n"));
        prompt.push_back(token_img_start);
        for (int i = 0; i < image_token_count; i++)
            prompt.push_back(token_img_context);
        prompt.push_back(token_img_end);
        append_tokens(prompt, tokenizer.encode("\n" + query));
        prompt.push_back(token_im_end);
        append_tokens(prompt, tokenizer.encode("\n"));
        prompt.push_back(token_im_start);
        append_tokens(prompt, tokenizer.encode("assistant\n"));
        return prompt;
    }

    bool forward_token_embedding(int token, const std::vector<float>* replacement, std::vector<float>& hidden) const
    {
        if (replacement)
        {
            hidden = *replacement;
            return true;
        }
        return load_embedding_row_from_table(embedding_table, token, hidden);
    }

    bool run_one(std::vector<float>& hidden, int position)
    {
        for (int i = 0; i < qwen_layers; i++)
        {
            if (!layer_loaded[i])
            {
                if (!load_qwen_layer(model_dir, i, layer_weights[i]))
                    return false;
                layer_loaded[i] = true;
            }
            std::vector<float> out;
            qwen_layer_forward_one(hidden, layer_weights[i], cache[i], position, out);
            hidden.swap(out);
        }
        return true;
    }

    bool run_block(std::vector<float>& hidden, int q_len, const std::vector<int>& positions)
    {
        for (int i = 0; i < qwen_layers; i++)
        {
            if (!layer_loaded[i])
            {
                if (!load_qwen_layer(model_dir, i, layer_weights[i]))
                    return false;
                layer_loaded[i] = true;
            }
            std::vector<float> out;
            qwen_layer_forward_block(hidden, q_len, layer_weights[i], cache[i], positions, out);
            hidden.swap(out);
        }
        return true;
    }

    int logits_argmax(std::vector<float> hidden)
    {
        std::vector<float> logits;
        logits_from_hidden(hidden, logits);
        int best = 0;
        float best_score = -INFINITY;
        for (int token = 0; token < qwen_vocab; token++)
        {
            const float score = logits[token];
            if (score > best_score)
            {
                best_score = score;
                best = token;
            }
        }
        fprintf(stderr, "locate_anything: argmax token=%d score=%f\n", best, best_score);
        return best;
    }

    void logits_from_hidden_cpu(std::vector<float> hidden, std::vector<float>& logits) const
    {
        rms_norm(hidden, final_norm);
        logits.resize(qwen_vocab);
        for (int token = 0; token < qwen_vocab; token++)
        {
            float score = 0.f;
            const uint16_t* wp = lm_head_table.data.data() + (size_t)token * qwen_hidden;
            for (int i = 0; i < qwen_hidden; i++)
                score += hidden[i] * fp16_to_float(wp[i]);
            logits[token] = score;
        }
    }

    bool logits_from_hidden_ncnn(const std::vector<float>& hidden, int rows, std::vector<float>& logits)
    {
        if (!lm_head_runtime_available)
            return false;

        ncnn::Mat in(qwen_hidden, rows);
        if (in.empty())
            return false;
        memcpy((float*)in, hidden.data(), (size_t)rows * qwen_hidden * sizeof(float));

        ncnn::Mat out;
        const int ret = run_blob(lm_head_runtime, in, out);
        if (ret != 0 || out.empty())
        {
            fprintf(stderr, "locate_anything: lm_head ncnn%s failed ret=%d empty=%d, fallback to raw CPU lm_head\n",
                    lm_head_runtime_vulkan ? " Vulkan" : "", ret, out.empty() ? 1 : 0);
            lm_head_runtime_available = false;
            return false;
        }

        logits.resize((size_t)rows * qwen_vocab);
        if (out.dims == 2 && out.w == qwen_vocab && out.h == rows)
        {
            memcpy(logits.data(), (const float*)out, (size_t)rows * qwen_vocab * sizeof(float));
            return true;
        }
        if (rows == 1 && out.w == qwen_vocab)
        {
            memcpy(logits.data(), (const float*)out, (size_t)qwen_vocab * sizeof(float));
            return true;
        }

        fprintf(stderr, "locate_anything: lm_head ncnn unexpected shape dims=%d w=%d h=%d c=%d, fallback to raw CPU lm_head\n",
                out.dims, out.w, out.h, out.c);
        lm_head_runtime_available = false;
        return false;
    }

    void logits_from_hidden(std::vector<float> hidden, std::vector<float>& logits)
    {
        std::vector<float> normed = hidden;
        rms_norm(normed, final_norm);
        if (logits_from_hidden_ncnn(normed, 1, logits))
            return;
        logits_from_hidden_cpu(hidden, logits);
    }

    void logits_from_hidden_block(std::vector<float> hidden, int rows, std::vector<float>& logits)
    {
        for (int i = 0; i < rows; i++)
        {
            float* row = hidden.data() + (size_t)i * qwen_hidden;
            std::vector<float> h(row, row + qwen_hidden);
            rms_norm(h, final_norm);
            memcpy(row, h.data(), qwen_hidden * sizeof(float));
        }
        if (lm_head_batch_runtime_enabled && logits_from_hidden_ncnn(hidden, rows, logits))
            return;

        logits.resize((size_t)rows * qwen_vocab);
        for (int i = 0; i < rows; i++)
        {
            const float* h = hidden.data() + (size_t)i * qwen_hidden;
            float* dst = logits.data() + (size_t)i * qwen_vocab;
            for (int token = 0; token < qwen_vocab; token++)
            {
                float score = 0.f;
                const uint16_t* wp = lm_head_table.data.data() + (size_t)token * qwen_hidden;
                for (int j = 0; j < qwen_hidden; j++)
                    score += h[j] * fp16_to_float(wp[j]);
                dst[token] = score;
            }
        }
    }

    bool prefill(const std::vector<int>& prompt, const std::vector<std::vector<float> >& image_embeds, int& next_id)
    {
        cache.assign(qwen_layers, KVCacheLayer());
        prefill_last_token = prompt.empty() ? token_im_end : prompt.back();
        std::vector<float> hidden;
        int image_index = 0;
        for (int i = 0; i < (int)prompt.size(); i++)
        {
            const std::vector<float>* replacement = 0;
            if (prompt[i] == token_img_context)
            {
                if (image_index >= (int)image_embeds.size())
                    return false;
                replacement = &image_embeds[image_index++];
            }
            if (!forward_token_embedding(prompt[i] == token_img_context ? token_endoftext : prompt[i], replacement, hidden))
                return false;
            if (!run_one(hidden, i))
                return false;
            fprintf(stderr, "locate_anything: prefill %d/%d token=%d\n", i + 1, (int)prompt.size(), prompt[i]);
        }
        next_id = logits_argmax(hidden);
        return next_id >= 0;
    }

    bool decode_greedy_ar(int first_id, int start_pos, int max_new_tokens, std::vector<int>& output)
    {
        int id = first_id;
        for (int step = 0; step < max_new_tokens; step++)
        {
            if (id == token_im_end || id == token_endoftext)
                break;
            output.push_back(id);
            std::vector<float> hidden;
            if (!load_embedding_row_from_table(embedding_table, id, hidden))
                return false;
            if (!run_one(hidden, start_pos + step))
                return false;
            id = logits_argmax(hidden);
            if (id < 0)
                return false;
            fprintf(stderr, "locate_anything: decode step=%d token=%d\n", step + 1, id);
        }
        return true;
    }

    bool decode_hybrid_ar_fallback(int first_id, int start_pos, int max_new_tokens, std::vector<int>& output)
    {
        (void)first_id;
        fprintf(stderr, "locate_anything: decode block_size=%d, using official hybrid MTP/AR path\n", locate_block_size);
        bool use_mtp = true;
        int switch_to_ar_count = 0;

        while ((int)output.size() < max_new_tokens)
        {
            MtpPattern pattern;
            if (use_mtp)
            {
                if (!decode_one_mtp_round(start_pos, max_new_tokens, output, pattern))
                    return false;
            }
            else
            {
                if (!decode_one_ar_round(start_pos, output, pattern))
                    return false;
            }

            if (pattern.type == "im_end")
                break;

            if (pattern.type == "error_box")
            {
                use_mtp = false;
                switch_to_ar_count++;
            }
            else if (pattern.type == "box_end_ar")
            {
                use_mtp = true;
            }

            fprintf(stderr, "locate_anything: decode mode=%s type=%s total=%d switch_to_ar=%d\n",
                    use_mtp ? "mtp" : "ar", pattern.type.c_str(), (int)output.size(), switch_to_ar_count);
        }
        return true;
    }

    std::string decode_text(const std::vector<int>& ids) const { return tokenizer.decode(ids); }

private:
    int cache_seq() const
    {
        return cache.empty() ? 0 : cache[0].seq;
    }

    void truncate_cache(int seq)
    {
        for (int i = 0; i < (int)cache.size(); i++)
        {
            if (cache[i].seq <= seq)
                continue;
            cache[i].seq = seq;
            cache[i].key.resize((size_t)seq * qwen_kv_heads * qwen_head_dim);
            cache[i].value.resize((size_t)seq * qwen_kv_heads * qwen_head_dim);
        }
    }

    bool token_ids_to_hidden_block(const std::vector<int>& ids, std::vector<float>& hidden) const
    {
        hidden.resize((size_t)ids.size() * qwen_hidden);
        for (int i = 0; i < (int)ids.size(); i++)
        {
            std::vector<float> row;
            if (!load_embedding_row_from_table(embedding_table, ids[i], row))
                return false;
            memcpy(hidden.data() + (size_t)i * qwen_hidden, row.data(), qwen_hidden * sizeof(float));
        }
        return true;
    }

    bool run_token_block(const std::vector<int>& ids, const std::vector<int>& positions, std::vector<float>& hidden)
    {
        if (ids.empty())
            return true;
        if (!token_ids_to_hidden_block(ids, hidden))
            return false;
        return run_block(hidden, (int)ids.size(), positions);
    }

    bool run_uncached_generated(int start_pos, const std::vector<int>& output, std::vector<float>& hidden)
    {
        const int first = cache_seq() - start_pos;
        if (first < 0 || first > (int)output.size())
            return false;
        std::vector<int> ids;
        std::vector<int> positions;
        for (int i = first; i < (int)output.size(); i++)
        {
            ids.push_back(output[i]);
            positions.push_back(start_pos + i);
        }
        return run_token_block(ids, positions, hidden);
    }

    std::vector<TokenProb> logits_topk_probs(const std::vector<float>& logits, int k, float* selected_probs, const std::vector<int>& selected_ids) const
    {
        std::vector<TokenLogit> top;
        top.reserve(k);
        float max_logit = -INFINITY;
        for (int i = 0; i < qwen_vocab; i++)
            max_logit = std::max(max_logit, logits[i]);
        double denom = 0.0;
        for (int i = 0; i < qwen_vocab; i++)
            denom += std::exp((double)logits[i] - max_logit);
        for (int i = 0; i < (int)selected_ids.size(); i++)
            selected_probs[i] = (float)(std::exp((double)logits[selected_ids[i]] - max_logit) / denom);

        for (int id = 0; id < qwen_vocab; id++)
        {
            if ((int)top.size() < k)
            {
                top.push_back(TokenLogit{id, logits[id]});
                if ((int)top.size() == k)
                    std::sort(top.begin(), top.end(), [](const TokenLogit& a, const TokenLogit& b) { return a.logit > b.logit; });
                continue;
            }
            if (logits[id] > top.back().logit)
            {
                top.back() = TokenLogit{id, logits[id]};
                for (int j = k - 1; j > 0 && top[j].logit > top[j - 1].logit; j--)
                    std::swap(top[j], top[j - 1]);
            }
        }

        std::vector<TokenProb> out;
        for (int i = 0; i < (int)top.size(); i++)
            out.push_back(TokenProb{top[i].id, (float)(std::exp((double)top[i].logit - max_logit) / denom)});
        return out;
    }

    static int first_valid_non_coord(const std::vector<TokenProb>& top)
    {
        for (size_t i = 0; i < top.size(); i++)
            if (!(top[i].id >= token_coord_start && top[i].id <= token_coord_end))
                return top[i].id;
        return -1;
    }

    bool decode_bbox_or_ref(const std::vector<std::vector<TokenProb> >& topk, const std::vector<std::vector<float> >& probs, std::vector<int>& decoded) const
    {
        if (probs[0][0] >= 0.7f)
        {
            if (probs[1][2] > 0.2f && probs[2][1] > 0.2f && probs[3][4] > 0.1f && probs[4][4] > 0.1f)
            {
                decoded = std::vector<int>{token_box_start, token_none, token_box_end, token_null, token_null, token_null};
                return true;
            }
        }

        const float end_score = probs[5][1] + probs[5][4] + probs[5][5];
        if (end_score >= 0.2f)
        {
            std::vector<int> coords;
            bool ok = true;
            for (int p = 1; p <= 4; p++)
            {
                int first_id = -1;
                float first_prob = 0.f;
                int valid_count = 0;
                int valid_min = 999999;
                int valid_max = -999999;
                for (size_t i = 0; i < topk[p].size(); i++)
                {
                    const int id = topk[p][i].id;
                    if (id >= token_coord_start && id <= token_coord_end)
                    {
                        if (first_id < 0)
                        {
                            first_id = id;
                            first_prob = topk[p][i].prob;
                        }
                        valid_count++;
                        valid_min = std::min(valid_min, id);
                        valid_max = std::max(valid_max, id);
                    }
                }
                if (first_id < 0)
                {
                    ok = false;
                    break;
                }
                if (first_prob < 0.9f && valid_count > 1 && (valid_max - valid_min) > 60)
                    coords.push_back(0);
                else
                    coords.push_back(first_id);
            }
            if (ok)
            {
                decoded.clear();
                decoded.push_back(token_box_start);
                decoded.insert(decoded.end(), coords.begin(), coords.end());
                decoded.push_back(token_box_end);
                return true;
            }
        }

        if (probs[0][3] >= 0.6f)
        {
            decoded.clear();
            decoded.push_back(token_ref_start);
            for (int p = 1; p < locate_block_size; p++)
            {
                int id = first_valid_non_coord(topk[p]);
                if (id < 0)
                    return false;
                decoded.push_back(id);
            }
            return true;
        }

        return false;
    }

    MtpPattern handle_pattern(const std::vector<int>& in) const
    {
        std::vector<int> x0 = in;
        MtpPattern out;
        if (x0.empty() || x0[0] == token_null || x0[0] == token_im_end)
        {
            out.type = "im_end";
            out.tokens.push_back(token_im_end);
            return out;
        }
        if (x0.size() >= 2 && x0[0] == token_box_start && x0[1] == token_none)
        {
            out.type = "empty_box";
            out.tokens = std::vector<int>{token_box_start, token_none, token_box_end};
            return out;
        }
        if (x0[0] == token_box_start)
        {
            int coord_ix = 1;
            for (int i = 1; i < 5 && i < (int)x0.size(); i++)
            {
                if (x0[i] >= token_coord_start && x0[i] <= token_coord_end)
                    coord_ix++;
                else
                    break;
            }
            if (coord_ix == 5 && (int)x0.size() > 5 && x0[5] == token_box_end)
            {
                out.type = "coord_box";
                out.tokens = x0;
                return out;
            }
            if (coord_ix == 3 && (int)x0.size() > 3 && x0[3] == token_box_end)
            {
                out.type = "point_box";
                out.tokens.assign(x0.begin(), x0.begin() + 4);
                return out;
            }
            out.type = "error_box";
            out.tokens.assign(x0.begin(), x0.begin() + std::min(coord_ix, (int)x0.size()));
            return out;
        }

        for (size_t i = 0; i < x0.size(); i++)
        {
            if (x0[i] == token_null)
            {
                x0.resize(i);
                break;
            }
        }
        if (x0.size() >= 2 && x0[x0.size() - 1] == token_ref_end && x0[x0.size() - 2] == token_ref_end)
            x0.pop_back();
        out.type = "ref_object";
        out.tokens = x0;
        return out;
    }

    bool decode_one_mtp_round(int start_pos, int max_new_tokens, std::vector<int>& output, MtpPattern& pattern)
    {
        const int generated_len = start_pos + (int)output.size();
        const int old_cache_seq = cache_seq();
        std::vector<int> ids;
        std::vector<int> positions;
        for (int abs_pos = old_cache_seq; abs_pos < generated_len; abs_pos++)
        {
            ids.push_back(output[abs_pos - start_pos]);
            positions.push_back(abs_pos);
        }
        const int last_token = output.empty() ? prefill_last_token : output.back();
        ids.push_back(last_token);
        positions.push_back(generated_len - 1);
        for (int i = 1; i < locate_block_size; i++)
        {
            ids.push_back(token_text_mask);
            positions.push_back(generated_len - 1 + i);
        }

        std::vector<float> hidden;
        if (!run_token_block(ids, positions, hidden))
            return false;

        truncate_cache(generated_len);

        const int q_len = (int)ids.size();
        std::vector<int> greedy(locate_block_size);
        std::vector<std::vector<TokenProb> > topk(locate_block_size);
        std::vector<std::vector<float> > selected_probs(locate_block_size);
        const std::vector<int> selected_ids = std::vector<int>{token_box_start, token_box_end, token_none, token_ref_start, token_null, token_im_end};
        std::vector<float> last_hidden(hidden.begin() + (size_t)(q_len - locate_block_size) * qwen_hidden, hidden.end());
        std::vector<float> block_logits;
        logits_from_hidden_block(last_hidden, locate_block_size, block_logits);
        for (int i = 0; i < locate_block_size; i++)
        {
            std::vector<float> logits(block_logits.begin() + (size_t)i * qwen_vocab,
                                      block_logits.begin() + ((size_t)i + 1) * qwen_vocab);
            float probs[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            topk[i] = logits_topk_probs(logits, 5, probs, selected_ids);
            selected_probs[i].assign(probs, probs + 6);
            greedy[i] = topk[i].empty() ? 0 : topk[i][0].id;
        }

        std::vector<int> decoded;
        const std::vector<int>& new_tokens = decode_bbox_or_ref(topk, selected_probs, decoded) ? decoded : greedy;
        pattern = handle_pattern(new_tokens);
        for (size_t i = 0; i < pattern.tokens.size(); i++)
            output.push_back(pattern.tokens[i]);
        return true;
    }

    bool decode_one_ar_round(int start_pos, std::vector<int>& output, MtpPattern& pattern)
    {
        std::vector<float> hidden;
        if (!run_uncached_generated(start_pos, output, hidden))
            return false;
        if (hidden.empty())
            return false;

        std::vector<float> last(hidden.end() - qwen_hidden, hidden.end());
        int id = logits_argmax(last);
        if (id < 0)
            return false;
        output.push_back(id);

        pattern.tokens = std::vector<int>{id};
        if (id == token_box_end)
            pattern.type = "box_end_ar";
        else if ((id >= token_coord_start && id <= token_coord_end) || id == token_none)
            pattern.type = "coord_ar";
        else if (id == token_im_end)
            pattern.type = "im_end";
        else
            pattern.type = "im_end";
        return true;
    }

    void append_chat_text(std::vector<int>& dst, const std::string& text) const
    {
        dst.push_back(token_im_start);
        append_tokens(dst, tokenizer.encode(text));
    }

    static void append_tokens(std::vector<int>& dst, const std::vector<int>& src)
    {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    std::string model_dir;
    HalfTensor final_norm;
    HalfTensor embedding_table;
    HalfTensor lm_head_table;
    ncnn::Net lm_head_runtime;
    bool lm_head_runtime_available;
    bool lm_head_runtime_vulkan;
    bool lm_head_batch_runtime_enabled;
    QwenTokenizer tokenizer;
    std::vector<KVCacheLayer> cache;
    std::vector<QwenLayerWeights> layer_weights;
    std::vector<int> layer_loaded;
    int prefill_last_token;
};

struct LocateResult
{
    std::string ref;
    float x1;
    float y1;
    float x2;
    float y2;
    bool has_box;
};

static std::vector<LocateResult> parse_locate_results(const std::vector<int>& ids, const LocateQwen& qwen, int image_w, int image_h)
{
    std::vector<LocateResult> results;
    std::string current_ref;
    for (size_t i = 0; i < ids.size(); i++)
    {
        if (ids[i] == token_ref_start)
        {
            std::vector<int> ref_ids;
            size_t j = i + 1;
            for (; j < ids.size() && ids[j] != token_ref_end; j++)
                ref_ids.push_back(ids[j]);
            current_ref = qwen.decode_text(ref_ids);
            i = j;
            continue;
        }

        if (ids[i] != token_box_start)
            continue;

        std::vector<int> coords;
        for (size_t j = i + 1; j < ids.size() && ids[j] != token_box_end; j++)
        {
            if (ids[j] >= token_coord_start && ids[j] <= token_coord_end)
                coords.push_back(ids[j] - token_coord_start);
        }
        if (coords.size() >= 4)
        {
            LocateResult r;
            r.ref = current_ref;
            r.x1 = coords[0] * image_w / 1000.f;
            r.y1 = coords[1] * image_h / 1000.f;
            r.x2 = coords[2] * image_w / 1000.f;
            r.y2 = coords[3] * image_h / 1000.f;
            r.has_box = true;
            results.push_back(r);
        }
    }
    return results;
}

static void print_locate_results(const std::vector<int>& ids, const LocateQwen& qwen, int image_w, int image_h)
{
    const std::vector<LocateResult> results = parse_locate_results(ids, qwen, image_w, image_h);
    for (size_t i = 0; i < results.size(); i++)
    {
        printf("result %d", (int)i);
        if (!results[i].ref.empty())
            printf(" ref=\"%s\"", results[i].ref.c_str());
        if (results[i].has_box)
            printf(" bbox=%.1f,%.1f,%.1f,%.1f", results[i].x1, results[i].y1, results[i].x2, results[i].y2);
        printf("\n");
    }
}

static int run_raw_qwen_one(int token, const std::string& model_dir)
{
    LocateQwen qwen;
    if (!qwen.load(model_dir))
        return -1;
    std::vector<float> hidden;
    if (!qwen.forward_token_embedding(token, 0, hidden))
        return -1;
    if (!qwen.run_one(hidden, 0))
        return -1;
    int next = qwen.logits_argmax(hidden);
    printf("%d\n", next);
    return next >= 0 ? 0 : -1;
}

static int run_raw_qwen_mtp(int token, const std::string& model_dir)
{
    LocateQwen qwen;
    if (!qwen.load(model_dir))
        return -1;
    std::vector<int> prompt;
    prompt.push_back(token);
    int next = -1;
    std::vector<std::vector<float> > no_image;
    if (!qwen.prefill(prompt, no_image, next))
        return -1;
    std::vector<int> output;
    if (!qwen.decode_hybrid_ar_fallback(next, 1, 1, output))
        return -1;
    printf("tokens:");
    for (size_t i = 0; i < output.size(); i++)
        printf(" %d", output[i]);
    printf("\n");
    return 0;
}

int main(int argc, char** argv)
{
    if (argc >= 3 && strcmp(argv[1], "--raw-qwen-one") == 0)
    {
        const int token = atoi(argv[2]);
        const std::string model_dir = argc >= 4 ? argv[3] : "examples/locate_everything";
        return run_raw_qwen_one(token, model_dir);
    }

    if (argc >= 3 && strcmp(argv[1], "--raw-qwen-mtp") == 0)
    {
        const int token = atoi(argv[2]);
        const std::string model_dir = argc >= 4 ? argv[3] : "examples/locate_everything";
        return run_raw_qwen_mtp(token, model_dir);
    }

    if (argc >= 3 && strcmp(argv[1], "--image-embed") == 0)
    {
        const char* imagepath = argv[2];
        const std::string model_dir = argc >= 4 ? argv[3] : "examples/locate_everything";
        cv::Mat bgr = cv::imread(imagepath, 1);
        if (bgr.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        std::vector<std::vector<float> > image_embeds;
        ImageResizeInfo info;
        if (run_image_embedding_dynamic(model_dir, bgr, image_embeds, info) != 0)
            return -1;
        printf("image_embeds %d x %d", (int)image_embeds.size(), qwen_hidden);
        if (!image_embeds.empty())
            for (int i = 0; i < 8 && i < (int)image_embeds[0].size(); i++)
                printf(" %.6f", image_embeds[0][i]);
        printf("\n");
        return 0;
    }

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s image.jpg [query] [max-new-tokens] [examples/locate_everything]\n", argv[0]);
        fprintf(stderr, "       %s --raw-qwen-one TOKEN [examples/locate_everything]\n", argv[0]);
        fprintf(stderr, "       %s --raw-qwen-mtp TOKEN [examples/locate_everything]\n", argv[0]);
        fprintf(stderr, "       %s --image-embed image.jpg [examples/locate_everything]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    std::string query = argc >= 3 ? argv[2] : "Locate all objects in the image.";
    int max_new_tokens = argc >= 4 ? atoi(argv[3]) : 8;
    const std::string model_dir = argc >= 5 ? argv[4] : "examples/locate_everything";

    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<std::vector<float> > image_embeds;
    ImageResizeInfo info;
    if (run_image_embedding_dynamic(model_dir, bgr, image_embeds, info) != 0)
    {
        fprintf(stderr, "locate_anything: image embedding failed\n");
        return -1;
    }
    fprintf(stderr, "locate_anything: dynamic image embeddings ready count=%d\n", (int)image_embeds.size());

    LocateQwen qwen;
    if (!qwen.load(model_dir))
    {
        fprintf(stderr, "locate_anything: qwen load failed\n");
        return -1;
    }

    std::vector<int> prompt = qwen.build_prompt(query, (int)image_embeds.size());
    int next_id = -1;
    if (!qwen.prefill(prompt, image_embeds, next_id))
    {
        fprintf(stderr, "locate_anything: qwen prefill failed\n");
        return -1;
    }

    std::vector<int> output;
    if (!qwen.decode_hybrid_ar_fallback(next_id, (int)prompt.size(), max_new_tokens, output))
    {
        fprintf(stderr, "locate_anything: qwen decode failed\n");
        return -1;
    }

    printf("tokens:");
    for (size_t i = 0; i < output.size(); i++)
        printf(" %d", output[i]);
    printf("\ntext: %s\n", qwen.decode_text(output).c_str());
    print_locate_results(output, qwen, bgr.cols, bgr.rows);
    return 0;
}
