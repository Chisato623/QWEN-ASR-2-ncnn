// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Qwen3-ASR-1.7B speech recognition implemented with ncnn library
//
// Put this executable in examples/Qwen3-ASR-1.7B or run it from that directory.
// Usage:
//   qwen_asr_1_7_b input.wav [language] [max-new-tokens]

#include "net.h"
#include "layer.h"
#include "layer_type.h"
#include "gpu.h"

#include <float.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

static const int token_endoftext = 151643;
static const int token_im_start = 151644;
static const int token_im_end = 151645;
static const int token_audio_start = 151669;
static const int token_audio_end = 151670;
static const int token_audio_pad = 151676;

static const int qwen_hidden_size = 2048;
static const int qwen_num_layers = 28;
static const int qwen_audio_hidden_size = 1024;
static const int qwen_num_kv_heads = 8;
static const int qwen_head_dim = 128;
static const int qwen_prefill_len = 512;
static const float qwen_rope_theta = 1000000.f;
static const int qwen_audio_max_frames = 3000;
static const int qwen_audio_chunk_frames = 100;
static const int qwen_audio_max_tokens = 390;

class PnnxExpression : public ncnn::Layer
{
public:
    PnnxExpression()
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

class AtenTo : public ncnn::Layer
{
public:
    AtenTo()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        if (bottom_blobs.empty())
            return -1;

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob = bottom_blobs[0].clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(PnnxExpression)
DEFINE_LAYER_CREATOR(AtenTo)

static void register_qwen3_asr_custom_layers(ncnn::Net& net)
{
    net.register_custom_layer("pnnx.Expression", PnnxExpression_layer_creator);
    net.register_custom_layer("aten::to", AtenTo_layer_creator);
}

static int load_ncnn_model(ncnn::Net& net, const char* param, const char* bin)
{
    int ret = net.load_param(param);
    if (ret != 0)
    {
        fprintf(stderr, "load_param %s failed %d\n", param, ret);
        return ret;
    }

    ret = net.load_model(bin);
    if (ret != 0)
    {
        fprintf(stderr, "load_model %s failed %d\n", bin, ret);
        return ret;
    }

    return 0;
}

static int qwen_asr_gpu_device()
{
#if NCNN_VULKAN
    static int selected_gpu_device = -2;
    if (selected_gpu_device != -2)
        return selected_gpu_device;

    int requested_gpu_device = -1;
    const char* env_gpu_device = getenv("NCNN_QWEN_ASR_GPU");
    if (env_gpu_device && env_gpu_device[0])
        requested_gpu_device = atoi(env_gpu_device);
    else
        fprintf(stderr, "qwen asr: vulkan disabled by default because it may produce incorrect text; set NCNN_QWEN_ASR_GPU=0 to test vulkan\n");

    if (requested_gpu_device < 0)
    {
        selected_gpu_device = -1;
        fprintf(stderr, "qwen asr: vulkan disabled by NCNN_QWEN_ASR_GPU=%d\n", requested_gpu_device);
        return selected_gpu_device;
    }

    const int gpu_count = ncnn::get_gpu_count();
    if (gpu_count <= 0)
    {
        selected_gpu_device = -1;
        fprintf(stderr, "qwen asr: no vulkan gpu found, fallback to cpu\n");
        return selected_gpu_device;
    }

    if (requested_gpu_device >= gpu_count)
    {
        fprintf(stderr, "qwen asr: requested gpu %d but only %d vulkan gpu(s) found, fallback to gpu 0\n", requested_gpu_device, gpu_count);
        requested_gpu_device = 0;
    }

    selected_gpu_device = requested_gpu_device;
    fprintf(stderr, "qwen asr: use vulkan gpu %d\n", selected_gpu_device);
    return selected_gpu_device;
#else
    static bool warned = false;
    if (!warned)
    {
        fprintf(stderr, "qwen asr: built without NCNN_VULKAN, use cpu\n");
        warned = true;
    }
    return -1;
#endif
}

static bool qwen_asr_gpu_fp16_enabled()
{
#if NCNN_VULKAN
    static int enabled = -1;
    if (enabled != -1)
        return enabled != 0;

    const char* env_fp16 = getenv("NCNN_QWEN_ASR_FP16");
    enabled = env_fp16 && env_fp16[0] && atoi(env_fp16) != 0 ? 1 : 0;
    if (!enabled)
        fprintf(stderr, "qwen asr: vulkan fp16 disabled by default for closer cpu text matching; set NCNN_QWEN_ASR_FP16=1 to test fp16\n");
    return enabled != 0;
#else
    return false;
#endif
}

static int qwen_asr_env_int(const char* name, int default_value)
{
    const char* value = getenv(name);
    if (!value || !value[0])
        return default_value;
    return atoi(value);
}

static void configure_qwen_asr_net(ncnn::Net& net, const char* name)
{
#if NCNN_VULKAN
    const int gpu_device = qwen_asr_gpu_device();
    if (gpu_device >= 0)
    {
        const bool use_fp16 = qwen_asr_gpu_fp16_enabled();
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = use_fp16;
        net.opt.use_fp16_storage = use_fp16;
        net.opt.use_fp16_arithmetic = use_fp16;
        net.opt.use_packing_layout = true;
        net.set_vulkan_device(gpu_device);
        fprintf(stderr, "qwen asr: enable vulkan for %s fp16=%d\n", name, use_fp16 ? 1 : 0);
    }
#else
    (void)net;
    (void)name;
#endif
}

static void configure_qwen_asr_cpu_net(ncnn::Net& net, const char* name)
{
    (void)net;
    (void)name;
}

static ncnn::Mat make_zero_kv_cache()
{
    ncnn::Mat cache(qwen_head_dim, qwen_prefill_len, qwen_num_kv_heads);
    cache.fill(0.f);
    return cache;
}

static void write_cache_row(ncnn::Mat& cache, int row, const ncnn::Mat& current)
{
    for (int q = 0; q < qwen_num_kv_heads; q++)
    {
        ncnn::Mat dst = cache.channel(q);
        const ncnn::Mat src = current.channel(q);
        memcpy(dst.row(row), src.row(0), qwen_head_dim * sizeof(float));
    }
}

class QwenTokenizer
{
public:
    bool load(const char* vocab_path, const char* merges_path)
    {
        generate_byte_decoder();

        std::string json;
        if (!read_file(vocab_path, json))
            return false;

        if (!parse_vocab_json(json))
            return false;

        FILE* fp = fopen(merges_path, "rb");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed\n", merges_path);
            return false;
        }

        char line[4096];
        int rank = 0;
        while (fgets(line, sizeof(line), fp))
        {
            std::string s = trim_newline(line);
            if (s.empty() || s[0] == '#')
                continue;
            size_t sp = s.find(' ');
            if (sp == std::string::npos)
                continue;
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

    void encode_piece(const std::string& text, std::vector<int>& ids) const
    {
        std::string bytes;
        bytes.reserve(text.size() * 2);

        for (unsigned char c : text)
        {
            append_utf8(bytes, byte_encoder[c]);
        }

        std::vector<std::string> word;
        for (size_t i = 0; i < bytes.size();)
        {
            int len = utf8_char_len((unsigned char)bytes[i]);
            word.push_back(bytes.substr(i, len));
            i += len;
        }

        for (;;)
        {
            int best_rank = INT_MAX;
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
        for (uint32_t cp : cps)
        {
            std::map<uint32_t, unsigned char>::const_iterator it = byte_decoder.find(cp);
            if (it != byte_decoder.end())
                out.push_back((char)it->second);
        }
        return out;
    }

private:
    static bool read_file(const char* path, std::string& content)
    {
        FILE* fp = fopen(path, "rb");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed\n", path);
            return false;
        }
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
        if (cp < 0x80)
            s.push_back((char)cp);
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
                while (i < text.size() && (text[i] == ' ' || text[i] == '\t'))
                    i++;
                if (i < text.size() && is_ascii_word((unsigned char)text[i]))
                {
                    while (i < text.size() && is_ascii_word((unsigned char)text[i]))
                        i++;
                }
                pieces.push_back(text.substr(start, i - start));
                continue;
            }

            if (c == '\r' || c == '\n')
            {
                while (i < text.size() && (text[i] == '\r' || text[i] == '\n'))
                    i++;
                pieces.push_back(text.substr(start, i - start));
                continue;
            }

            if (is_ascii_word(c))
            {
                while (i < text.size() && is_ascii_word((unsigned char)text[i]))
                    i++;
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
            if (len == 2 && i + 1 < s.size())
                cp = ((s[i] & 0x1f) << 6) | (s[i + 1] & 0x3f);
            else if (len == 3 && i + 2 < s.size())
                cp = ((s[i] & 0x0f) << 12) | ((s[i + 1] & 0x3f) << 6) | (s[i + 2] & 0x3f);
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
        {
            if (!used.count(b))
            {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
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
            if (json[i] != '"')
            {
                i++;
                continue;
            }

            size_t k0 = ++i;
            std::string key;
            while (i < json.size() && json[i] != '"')
            {
                if (json[i] == '\\' && i + 1 < json.size())
                    i++;
                key.push_back(json[i++]);
            }
            i++;
            while (i < json.size() && json[i] != ':') i++;
            i++;
            int id = atoi(json.c_str() + i);
            (void)k0;
            vocab[key] = id;
            if (id >= (int)id_to_token.size())
                id_to_token.resize(id + 1);
            id_to_token[id] = key;
        }
        return !vocab.empty();
    }

private:
    uint32_t byte_encoder[256];
    std::map<uint32_t, unsigned char> byte_decoder;
    std::map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
};

static int load_wav_samples(const char* wavpath, std::vector<short>& samples)
{
    FILE* fp = fopen(wavpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "open %s failed\n", wavpath);
        return -1;
    }

#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

    PACK(struct wav_header {
        char riff[4];
        uint32_t chunk_size;
        char wave[4];
        char fmt[4];
        uint32_t subchunk1_size;
        uint16_t audio_format;
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
        char data[4];
        uint32_t data_size;
    });

    wav_header header;
    if (fread(&header, sizeof(wav_header), 1, fp) != 1)
    {
        fclose(fp);
        return -1;
    }

    if (memcmp(header.riff, "RIFF", 4) != 0 || memcmp(header.wave, "WAVE", 4) != 0 ||
        memcmp(header.fmt, "fmt ", 4) != 0 || memcmp(header.data, "data", 4) != 0 ||
        header.subchunk1_size != 16 || header.audio_format != 1 || header.num_channels != 1 ||
        header.sample_rate != 16000 || header.bits_per_sample != 16)
    {
        fprintf(stderr, "%s is not pcm s16le 16k mono wav\n", wavpath);
        fclose(fp);
        return -1;
    }

    samples.resize(header.data_size / sizeof(short));
    fread(samples.data(), 1, header.data_size, fp);
    fclose(fp);
    return 0;
}

static std::string csv_escape(const std::string& s)
{
    bool need_quote = false;
    for (size_t i = 0; i < s.size(); i++)
    {
        if (s[i] == '"' || s[i] == ',' || s[i] == '\n' || s[i] == '\r')
        {
            need_quote = true;
            break;
        }
    }

    if (!need_quote)
        return s;

    std::string out = "\"";
    for (size_t i = 0; i < s.size(); i++)
    {
        if (s[i] == '"')
            out += "\"\"";
        else
            out += s[i];
    }
    out += "\"";
    return out;
}

static int split_tsv_line(const std::string& line, std::string& audio_path, std::string& relpath)
{
    size_t tab = line.find('\t');
    if (tab == std::string::npos)
        return -1;
    audio_path = line.substr(0, tab);
    relpath = line.substr(tab + 1);
    while (!relpath.empty() && (relpath.back() == '\n' || relpath.back() == '\r'))
        relpath.pop_back();
    return 0;
}

static std::string trim_ascii_space(const std::string& s)
{
    size_t begin = 0;
    while (begin < s.size() && isspace((unsigned char)s[begin]))
        begin++;

    size_t end = s.size();
    while (end > begin && isspace((unsigned char)s[end - 1]))
        end--;

    return s.substr(begin, end - begin);
}

static bool starts_with_ascii_case_insensitive(const std::string& s, const char* prefix)
{
    for (size_t i = 0; prefix[i]; i++)
    {
        if (i >= s.size())
            return false;
        if (tolower((unsigned char)s[i]) != tolower((unsigned char)prefix[i]))
            return false;
    }
    return true;
}

static const char* const supported_languages[] = {
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
};

static std::string trim_language_separator(const std::string& s)
{
    size_t begin = 0;
    while (begin < s.size())
    {
        unsigned char c = (unsigned char)s[begin];
        if (!isspace(c) && c != ':' && c != ',' && c != '.' && c != '-' && c != '_')
            break;
        begin++;
    }
    return trim_ascii_space(s.substr(begin));
}

static std::string remove_asr_text_marker(const std::string& text)
{
    std::string out = trim_ascii_space(text);
    while (starts_with_ascii_case_insensitive(out, "<asr_text>"))
        out = trim_ascii_space(out.substr(strlen("<asr_text>")));
    return out;
}

static std::string remove_language_header(const std::string& text)
{
    std::string out = remove_asr_text_marker(text);

    while (starts_with_ascii_case_insensitive(out, "language "))
    {
        std::string rest = trim_ascii_space(out.substr(strlen("language ")));
        bool matched = false;
        for (const char* lang : supported_languages)
        {
            if (starts_with_ascii_case_insensitive(rest, lang))
            {
                out = remove_asr_text_marker(trim_language_separator(rest.substr(strlen(lang))));
                matched = true;
                break;
            }
        }
        if (matched)
            continue;

        size_t line_end = out.find_first_of("\r\n");
        if (line_end == std::string::npos)
            return std::string();

        out = trim_ascii_space(out.substr(line_end + 1));
    }

    return out;
}

static int get_audio_token_count_from_feature_len(int feature_len)
{
    if (feature_len <= 0)
        return 0;

    feature_len = std::min(feature_len, qwen_audio_max_frames);
    const int full_chunks = feature_len / qwen_audio_chunk_frames;
    const int leave = feature_len % qwen_audio_chunk_frames;
    int output_lengths = full_chunks * 13;
    if (leave != 0)
    {
        const int feat_lengths = (leave - 1) / 2 + 1;
        output_lengths += (((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1);
    }
    return std::min(output_lengths, qwen_audio_max_tokens);
}

static int get_audio_chunk_aftercnn_length(int chunk_length)
{
    if (chunk_length <= 0)
        return 0;

    const int input_lengths_leave = chunk_length % 100;
    const int feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    return ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (chunk_length / 100) * 13;
}

static void fill_rope_cache_row(float* cosptr, float* sinptr, int position)
{
    const int half_head_dim = qwen_head_dim / 2;
    for (int j = 0; j < qwen_head_dim; j++)
    {
        const int freq_index = j % half_head_dim;
        float inv_freq = powf(qwen_rope_theta, -(float)(2 * freq_index) / qwen_head_dim);
        float v = position * inv_freq;
        cosptr[j] = cosf(v);
        sinptr[j] = sinf(v);
    }
}

static float audio_position_value(int position, int channel)
{
    const int half_dim = qwen_audio_hidden_size / 2;
    const int freq_index = channel < half_dim ? channel : channel - half_dim;
    const float log_timescale_increment = logf(10000.f) / (half_dim - 1);
    const float inv_timescale = expf(-log_timescale_increment * freq_index);
    const float scaled_time = position * inv_timescale;
    return channel < half_dim ? sinf(scaled_time) : cosf(scaled_time);
}

static bool extract_language_header(const std::string& text, std::string& language, std::string& body)
{
    std::string out = trim_ascii_space(text);
    if (!starts_with_ascii_case_insensitive(out, "language "))
        return false;

    std::string rest = trim_ascii_space(out.substr(strlen("language ")));
    for (const char* lang : supported_languages)
    {
        if (starts_with_ascii_case_insensitive(rest, lang))
        {
            language = lang;
            body = remove_language_header(out);
            return true;
        }
    }

    size_t line_end = out.find_first_of("\r\n");
    std::string first_line = line_end == std::string::npos ? out : out.substr(0, line_end);
    language = trim_ascii_space(first_line.substr(strlen("language ")));
    body = line_end == std::string::npos ? std::string() : remove_language_header(out.substr(line_end + 1));
    return !language.empty();
}

static bool is_decimal_number(const char* s)
{
    if (!s || !s[0])
        return false;

    for (const char* p = s; *p; p++)
    {
        if (!isdigit((unsigned char)*p))
            return false;
    }
    return true;
}

static bool load_f32(const char* path, std::vector<float>& data)
{
    FILE* fp = fopen(path, "rb");
    if (!fp)
        return false;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    rewind(fp);
    data.resize(len / sizeof(float));
    fread(data.data(), sizeof(float), data.size(), fp);
    fclose(fp);
    return true;
}

class Qwen3ASR
{
public:
    int load();
    int transcribe(const std::vector<short>& samples, const char* language, int max_new_tokens, std::string& text, std::string& detected_language);

private:
    int extract_fbank(const std::vector<short>& samples, ncnn::Mat& features, int& feature_len) const;
    int run_audio(const ncnn::Mat& features, int audio_token_count, ncnn::Mat& audio_embeds) const;
    int decode(const ncnn::Mat& audio_embeds, const char* language, int max_new_tokens, std::string& text) const;
    int token_embed(int token, ncnn::Mat& hidden) const;
    int run_text_prefill(const ncnn::Mat& hidden, int valid_tokens, ncnn::Mat& logits, std::vector<ncnn::Mat>& cache) const;
    int run_text_step(const ncnn::Mat& hidden, int position, std::vector<ncnn::Mat>& cache, ncnn::Mat& logits) const;

private:
    ncnn::Net audio_cnn;
    ncnn::Net audio_transformer;
    ncnn::Net audio_proj;
    ncnn::Net text_embed;
    ncnn::Net text_norm;
    ncnn::Net lm_head;
    ncnn::Net decoder_prefill_layers[qwen_num_layers];
    ncnn::Net decoder_step_layers[qwen_num_layers];
    QwenTokenizer tokenizer;
    std::vector<float> mel_filters;
};

int Qwen3ASR::load()
{
    configure_qwen_asr_net(audio_cnn, "audio_cnn");
    configure_qwen_asr_net(audio_transformer, "audio_transformer");
    configure_qwen_asr_net(audio_proj, "audio_proj");

    const int text_gpu = qwen_asr_env_int("NCNN_QWEN_ASR_TEXT_GPU", 0);
    const int text_embed_gpu = qwen_asr_env_int("NCNN_QWEN_ASR_TEXT_EMBED_GPU", text_gpu);
    const int text_head_gpu = qwen_asr_env_int("NCNN_QWEN_ASR_TEXT_HEAD_GPU", text_gpu);
    const int prefill_gpu_layers = std::min(qwen_num_layers, std::max(0, qwen_asr_env_int("NCNN_QWEN_ASR_PREFILL_GPU_LAYERS", text_gpu ? qwen_num_layers : 0)));
    const int step_gpu_layers = std::min(qwen_num_layers, std::max(0, qwen_asr_env_int("NCNN_QWEN_ASR_STEP_GPU_LAYERS", text_gpu ? qwen_num_layers : 0)));

    if (text_gpu || text_embed_gpu || text_head_gpu || prefill_gpu_layers || step_gpu_layers)
        fprintf(stderr, "qwen asr: text vulkan requested embed=%d head=%d prefill_layers=%d step_layers=%d\n", text_embed_gpu, text_head_gpu, prefill_gpu_layers, step_gpu_layers);
    else
        fprintf(stderr, "qwen asr: text vulkan disabled by default for stable recognition; set NCNN_QWEN_ASR_TEXT_GPU=1 to test it\n");

    if (text_embed_gpu)
        configure_qwen_asr_net(text_embed, "text_embed");
    else
        configure_qwen_asr_cpu_net(text_embed, "text_embed");

    if (text_head_gpu)
    {
        configure_qwen_asr_net(text_norm, "text_norm");
        configure_qwen_asr_net(lm_head, "lm_head");
    }
    else
    {
        configure_qwen_asr_cpu_net(text_norm, "text_norm");
        configure_qwen_asr_cpu_net(lm_head, "lm_head");
    }

    for (int i = 0; i < qwen_num_layers; i++)
    {
        char name[64];
        sprintf(name, "decoder_prefill_layer_%02d", i);
        if (i < prefill_gpu_layers)
            configure_qwen_asr_net(decoder_prefill_layers[i], name);
        else
            configure_qwen_asr_cpu_net(decoder_prefill_layers[i], name);
        sprintf(name, "decoder_step_layer_%02d", i);
        if (i < step_gpu_layers)
            configure_qwen_asr_net(decoder_step_layers[i], name);
        else
            configure_qwen_asr_cpu_net(decoder_step_layers[i], name);
    }

    register_qwen3_asr_custom_layers(audio_cnn);
    register_qwen3_asr_custom_layers(audio_transformer);
    register_qwen3_asr_custom_layers(audio_proj);
    register_qwen3_asr_custom_layers(text_embed);
    register_qwen3_asr_custom_layers(text_norm);
    register_qwen3_asr_custom_layers(lm_head);
    for (int i = 0; i < qwen_num_layers; i++)
    {
        register_qwen3_asr_custom_layers(decoder_prefill_layers[i]);
        register_qwen3_asr_custom_layers(decoder_step_layers[i]);
    }

    if (load_ncnn_model(audio_cnn, "qwen3_asr_1_7b_audio_cnn.ncnn.param", "qwen3_asr_1_7b_audio_cnn.ncnn.bin") != 0) return -1;
    if (load_ncnn_model(audio_transformer, "qwen3_asr_1_7b_audio_transformer.ncnn.param", "qwen3_asr_1_7b_audio_transformer.ncnn.bin") != 0) return -1;
    if (load_ncnn_model(audio_proj, "qwen3_asr_1_7b_audio_proj.ncnn.param", "qwen3_asr_1_7b_audio_proj.ncnn.bin") != 0) return -1;
    if (load_ncnn_model(text_embed, "qwen3_asr_1_7b_text_embed.ncnn.param", "qwen3_asr_1_7b_text_embed.ncnn.bin") != 0) return -1;
    if (load_ncnn_model(text_norm, "qwen3_asr_1_7b_text_norm.ncnn.param", "qwen3_asr_1_7b_text_norm.ncnn.bin") != 0) return -1;
    if (load_ncnn_model(lm_head, "qwen3_asr_1_7b_lm_head.ncnn.param", "qwen3_asr_1_7b_lm_head.ncnn.bin") != 0) return -1;

    for (int i = 0; i < qwen_num_layers; i++)
    {
        char param[128];
        char bin[128];
        sprintf(param, "qwen3_asr_1_7b_text_decoder_prefill_layer_%02d.ncnn.param", i);
        sprintf(bin, "qwen3_asr_1_7b_text_decoder_prefill_layer_%02d.ncnn.bin", i);
        if (load_ncnn_model(decoder_prefill_layers[i], param, bin) != 0)
            return -1;

        sprintf(param, "qwen3_asr_1_7b_text_decoder_step_layer_%02d.ncnn.param", i);
        sprintf(bin, "qwen3_asr_1_7b_text_decoder_step_layer_%02d.ncnn.bin", i);
        if (load_ncnn_model(decoder_step_layers[i], param, bin) != 0)
            return -1;
    }

    if (!load_f32("mel_filters.f32.bin", mel_filters))
    {
        fprintf(stderr, "load mel_filters.f32.bin failed\n");
        return -1;
    }
    if (!tokenizer.load("vocab.json", "merges.txt"))
    {
        fprintf(stderr, "load tokenizer failed\n");
        return -1;
    }

    return 0;
}

int Qwen3ASR::extract_fbank(const std::vector<short>& samples, ncnn::Mat& features, int& feature_len) const
{
    const int n_samples = 480000;
    feature_len = std::min(qwen_audio_max_frames, (int)((samples.size() + 159) / 160));

    ncnn::Mat waveform(n_samples);
    waveform.fill(0.f);
    for (size_t i = 0; i < samples.size() && i < (size_t)n_samples; i++)
        waveform[i] = samples[i] / 32768.f;

    ncnn::Layer* spectrogram = ncnn::create_layer_cpu("Spectrogram");
    ncnn::ParamDict pd;
    pd.set(0, 400);
    pd.set(1, 2);
    pd.set(2, 160);
    pd.set(3, 400);
    pd.set(4, 1);
    pd.set(5, 1);
    pd.set(6, 2);
    pd.set(7, 0);
    pd.set(8, 1);
    spectrogram->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    ncnn::Mat spec;
    int ret = spectrogram->forward(waveform, spec, opt);
    delete spectrogram;
    if (ret != 0 || spec.empty())
    {
        fprintf(stderr, "Spectrogram failed %d\n", ret);
        return -1;
    }
    features.create(qwen_audio_max_frames, 128);
    for (int m = 0; m < 128; m++)
    {
        float* outptr = features.row(m);
        for (int t = 0; t < qwen_audio_max_frames; t++)
        {
            double v = 0.0;
            if (t < spec.w)
            {
                const int bins = std::min(201, spec.h);
                for (int f = 0; f < bins; f++)
                    v += mel_filters[f * 128 + m] * spec.row(f)[t];
            }
            outptr[t] = log10f(std::max((float)v, 1e-10f));
        }
    }

    float maxv = -FLT_MAX;
    for (int i = 0; i < features.total(); i++)
        maxv = std::max(maxv, features[i]);
    const float floorv = maxv - 8.f;
    for (int i = 0; i < features.total(); i++)
        features[i] = (std::max(features[i], floorv) + 4.f) / 4.f;

    return 0;
}

int Qwen3ASR::run_audio(const ncnn::Mat& features, int audio_token_count, ncnn::Mat& audio_embeds) const
{
    ncnn::Mat audio_states(qwen_audio_hidden_size, qwen_audio_max_tokens);
    audio_states.fill(0.f);
    ncnn::Mat audio_mask(qwen_audio_max_tokens, qwen_audio_max_tokens, 1, 1);
    audio_mask.fill(-1e30f);

    int state_row = 0;
    int segment_begin = 0;
    for (int chunk_index = 0; chunk_index < 30; chunk_index++)
    {
        const int feature_begin = chunk_index * 100;
        const int raw_chunk_length = std::min(100, std::max(0, qwen_audio_max_frames - feature_begin));
        const int chunk_aftercnn_len = get_audio_chunk_aftercnn_length(raw_chunk_length);

        ncnn::Mat chunk(100, 128, 1);
        chunk.fill(0.f);
        for (int m = 0; m < 128; m++)
            memcpy(chunk.row(m), features.row(m) + chunk_index * 100, 100 * sizeof(float));

        ncnn::Mat cnn_out;
        ncnn::Extractor ex0 = audio_cnn.create_extractor();
        int ret = ex0.input("in0", chunk);
        if (ret != 0)
        {
            fprintf(stderr, "audio_cnn input failed %d chunk=%d\n", ret, chunk_index);
            return ret;
        }
        ret = ex0.extract("out0", cnn_out);
        if (ret != 0 || cnn_out.empty())
        {
            fprintf(stderr, "audio_cnn extract failed %d chunk=%d\n", ret, chunk_index);
            return ret != 0 ? ret : -1;
        }

        const int copy_rows = std::min(cnn_out.h, qwen_audio_max_tokens - state_row);
        for (int i = 0; i < copy_rows; i++)
        {
            const float* src = cnn_out.row(i);
            float* dst = audio_states.row(state_row + i);
            const int global_pos = state_row + i;
            const int local_pos = i;
            for (int c = 0; c < qwen_audio_hidden_size; c++)
                dst[c] = src[c] + audio_position_value(local_pos, c) - audio_position_value(global_pos, c);
        }
        state_row += copy_rows;

        const int segment_end = std::min(state_row, qwen_audio_max_tokens);
        for (int y = segment_begin; y < segment_end; y++)
        {
            float* mask_row = audio_mask.row(y);
            for (int x = segment_begin; x < segment_end; x++)
                mask_row[x] = 0.f;
        }
        segment_begin = segment_end;
    }
    ncnn::Mat transformer_out;
    ncnn::Extractor ex1 = audio_transformer.create_extractor();
    int ret = ex1.input("in0", audio_states);
    ret |= ex1.input("in1", audio_mask);
    if (ret != 0)
    {
        fprintf(stderr, "audio_transformer input failed %d\n", ret);
        return ret;
    }
    ret = ex1.extract("out0", transformer_out);
    if (ret != 0 || transformer_out.empty())
    {
        fprintf(stderr, "audio_transformer extract failed %d\n", ret);
        return ret != 0 ? ret : -1;
    }
    ncnn::Extractor ex2 = audio_proj.create_extractor();
    ret = ex2.input("in0", transformer_out);
    if (ret != 0)
    {
        fprintf(stderr, "audio_proj input failed %d\n", ret);
        return ret;
    }
    ret = ex2.extract("out0", audio_embeds);
    if (ret != 0 || audio_embeds.empty())
    {
        fprintf(stderr, "audio_proj extract failed %d\n", ret);
        return ret != 0 ? ret : -1;
    }
    if (audio_token_count > 0 && audio_token_count < audio_embeds.h)
        audio_embeds = audio_embeds.row_range(0, audio_token_count).clone();
    return 0;
}

int Qwen3ASR::token_embed(int token, ncnn::Mat& hidden) const
{
    ncnn::Mat input(1);
    ((int*)input)[0] = token;
    ncnn::Extractor ex = text_embed.create_extractor();
    ex.input("in0", input);
    return ex.extract("out0", hidden);
}

int Qwen3ASR::run_text_prefill(const ncnn::Mat& hidden0, int valid_tokens, ncnn::Mat& logits, std::vector<ncnn::Mat>& cache) const
{
    ncnn::Mat hidden = hidden0;
    cache.resize(qwen_num_layers * 2);

    ncnn::Mat cos_cache(qwen_head_dim, qwen_prefill_len);
    ncnn::Mat sin_cache(qwen_head_dim, qwen_prefill_len);
    for (int p = 0; p < qwen_prefill_len; p++)
    {
        float* cosptr = cos_cache.row(p);
        float* sinptr = sin_cache.row(p);
        fill_rope_cache_row(cosptr, sinptr, p);
    }

    ncnn::Mat mask(qwen_prefill_len, qwen_prefill_len, 1);
    for (int y = 0; y < qwen_prefill_len; y++)
    {
        float* ptr = mask.row(y);
        for (int x = 0; x < qwen_prefill_len; x++)
        {
            ptr[x] = (x > y || x >= valid_tokens) ? -1e30f : 0.f;
        }
    }

    for (int i = 0; i < qwen_num_layers; i++)
    {
        ncnn::Extractor ex = decoder_prefill_layers[i].create_extractor();
        int ret = 0;
        ret |= ex.input("in0", hidden);
        ret |= ex.input("in1", cos_cache);
        ret |= ex.input("in2", sin_cache);
        ret |= ex.input("in3", mask);
        if (ret != 0)
        {
            fprintf(stderr, "prefill layer %d input failed %d\n", i, ret);
            return ret;
        }

        ret = ex.extract("out1", cache[i * 2]);
        ret |= ex.extract("out2", cache[i * 2 + 1]);
        ret |= ex.extract("out0", hidden);
        if (ret != 0 || hidden.empty())
        {
            fprintf(stderr, "prefill layer %d extract failed %d\n", i, ret);
            return ret != 0 ? ret : -1;
        }
    }

    ncnn::Mat last_hidden = hidden.row_range(valid_tokens - 1, 1).clone();

    ncnn::Mat normed;
    ncnn::Extractor exn = text_norm.create_extractor();
    int ret = exn.input("in0", last_hidden);
    if (ret != 0)
        return ret;
    ret = exn.extract("out0", normed);
    if (ret != 0 || normed.empty())
    {
        fprintf(stderr, "text_norm prefill extract failed %d\n", ret);
        return ret != 0 ? ret : -1;
    }

    ncnn::Extractor exl = lm_head.create_extractor();
    ret = exl.input("in0", normed);
    if (ret != 0)
        return ret;
    ret = exl.extract("out0", logits);
    if (ret != 0 || logits.empty())
    {
        fprintf(stderr, "lm_head prefill extract failed %d\n", ret);
        return ret != 0 ? ret : -1;
    }
    logits = logits.reshape(logits.w);

    return 0;
}

int Qwen3ASR::run_text_step(const ncnn::Mat& hidden0, int position, std::vector<ncnn::Mat>& cache, ncnn::Mat& logits) const
{
    ncnn::Mat hidden = hidden0.reshape(qwen_hidden_size, 1);

    const int cache_pos = position % qwen_prefill_len;
    const int visible = std::min(position, qwen_prefill_len);

    for (int i = 0; i < qwen_num_layers; i++)
    {
        ncnn::Mat cos_cache(qwen_head_dim, 1);
        ncnn::Mat sin_cache(qwen_head_dim, 1);
        fill_rope_cache_row(cos_cache, sin_cache, position);

        ncnn::Mat mask(qwen_prefill_len + 1, 1, 1);
        float* maskptr = mask;
        for (int j = 0; j < qwen_prefill_len; j++)
            maskptr[j] = j < visible ? 0.f : -1e30f;
        maskptr[qwen_prefill_len] = 0.f;

        ncnn::Extractor ex = decoder_step_layers[i].create_extractor();
        int ret = 0;
        ret |= ex.input("in0", hidden);
        ret |= ex.input("in1", cos_cache);
        ret |= ex.input("in2", sin_cache);
        ret |= ex.input("in3", mask);
        ret |= ex.input("in4", cache[i * 2]);
        ret |= ex.input("in5", cache[i * 2 + 1]);
        if (ret != 0)
        {
            fprintf(stderr, "step layer %d input failed %d pos=%d\n", i, ret, position);
            return ret;
        }

        ncnn::Mat current_k;
        ncnn::Mat current_v;
        ret = ex.extract("out1", current_k);
        ret |= ex.extract("out2", current_v);
        ret |= ex.extract("out0", hidden);
        if (ret != 0 || hidden.empty())
        {
            fprintf(stderr, "step layer %d extract failed %d pos=%d\n", i, ret, position);
            return ret != 0 ? ret : -1;
        }
        write_cache_row(cache[i * 2], cache_pos, current_k);
        write_cache_row(cache[i * 2 + 1], cache_pos, current_v);
    }

    ncnn::Mat normed;
    ncnn::Extractor exn = text_norm.create_extractor();
    int ret = exn.input("in0", hidden);
    if (ret != 0)
        return ret;
    ret = exn.extract("out0", normed);
    if (ret != 0 || normed.empty())
    {
        fprintf(stderr, "text_norm extract failed %d\n", ret);
        return ret != 0 ? ret : -1;
    }

    ncnn::Extractor exl = lm_head.create_extractor();
    ret = exl.input("in0", normed);
    if (ret != 0)
        return ret;
    ret = exl.extract("out0", logits);
    if (ret != 0 || logits.empty())
    {
        fprintf(stderr, "lm_head extract failed %d\n", ret);
        return ret != 0 ? ret : -1;
    }
    logits = logits.reshape(logits.w);

    return 0;
}

int Qwen3ASR::decode(const ncnn::Mat& audio_embeds, const char* language, int max_new_tokens, std::string& text) const
{
    std::string system_prompt;

    std::vector<int> prompt;
    prompt.push_back(token_im_start);
    std::vector<int> v = tokenizer.encode("system\n" + system_prompt);
    prompt.insert(prompt.end(), v.begin(), v.end());
    prompt.push_back(token_im_end);
    v = tokenizer.encode("\n");
    prompt.insert(prompt.end(), v.begin(), v.end());
    prompt.push_back(token_im_start);
    v = tokenizer.encode("user\n");
    prompt.insert(prompt.end(), v.begin(), v.end());
    prompt.push_back(token_audio_start);

    for (int i = 0; i < audio_embeds.h; i++)
        prompt.push_back(token_audio_pad);

    prompt.push_back(token_audio_end);
    prompt.push_back(token_im_end);
    v = tokenizer.encode("\n");
    prompt.insert(prompt.end(), v.begin(), v.end());
    prompt.push_back(token_im_start);
    v = tokenizer.encode("assistant\n");
    prompt.insert(prompt.end(), v.begin(), v.end());
    if (language && language[0])
    {
        std::string force_language = "language ";
        force_language += language;
        force_language += "<asr_text>";
        v = tokenizer.encode(force_language);
        prompt.insert(prompt.end(), v.begin(), v.end());
    }

    if ((int)prompt.size() > qwen_prefill_len)
    {
        fprintf(stderr, "prompt too long: %d > %d\n", (int)prompt.size(), qwen_prefill_len);
        return -1;
    }

    ncnn::Mat prefill_hidden(qwen_hidden_size, qwen_prefill_len);
    ncnn::Mat pad_hidden;
    if (token_embed(token_endoftext, pad_hidden) != 0)
        return -1;
    for (int i = 0; i < qwen_prefill_len; i++)
        memcpy(prefill_hidden.row(i), pad_hidden.row(0), qwen_hidden_size * sizeof(float));
    int audio_index = 0;
    for (int i = 0; i < (int)prompt.size(); i++)
    {
        if (prompt[i] == token_audio_pad && audio_index < audio_embeds.h)
        {
            memcpy(prefill_hidden.row(i), audio_embeds.row(audio_index), qwen_hidden_size * sizeof(float));
            audio_index++;
            continue;
        }

        ncnn::Mat hidden;
        if (token_embed(prompt[i], hidden) != 0)
        {
            fprintf(stderr, "token_embed failed id=%d\n", prompt[i]);
            return -1;
        }
        memcpy(prefill_hidden.row(i), hidden.row(0), qwen_hidden_size * sizeof(float));
    }

    std::vector<ncnn::Mat> cache;
    ncnn::Mat logits;
    if (run_text_prefill(prefill_hidden, (int)prompt.size(), logits, cache) != 0)
        return -1;

    int pos = (int)prompt.size();

    std::vector<int> output_ids;
    int next_id = 0;
    for (int step = 0; step < max_new_tokens; step++)
    {
        next_id = (int)(std::max_element((const float*)logits, (const float*)logits + logits.w) - (const float*)logits);
        if (next_id == token_im_end || next_id == token_endoftext)
            break;
        output_ids.push_back(next_id);

        ncnn::Mat hidden;
        if (token_embed(next_id, hidden) != 0)
            return -1;
        if (run_text_step(hidden, pos++, cache, logits) != 0)
            return -1;
    }

    text = tokenizer.decode(output_ids);
    return 0;
}

int Qwen3ASR::transcribe(const std::vector<short>& samples, const char* language, int max_new_tokens, std::string& text, std::string& detected_language)
{
    ncnn::Mat features;
    int feature_len = 0;
    if (extract_fbank(samples, features, feature_len) != 0)
        return -1;

    const int audio_token_count = get_audio_token_count_from_feature_len(feature_len);
    ncnn::Mat audio_embeds;
    if (run_audio(features, audio_token_count, audio_embeds) != 0)
        return -1;

    detected_language.clear();

    if (language && language[0])
    {
        detected_language = language;
        int ret = decode(audio_embeds, language, max_new_tokens, text);
        if (ret != 0)
            return ret;
        text = remove_language_header(text);
        if (!text.empty())
            return 0;

        std::string raw_text;
        ret = decode(audio_embeds, "", max_new_tokens, raw_text);
        if (ret != 0)
            return ret;

        std::string fallback_language;
        std::string fallback_body;
        if (extract_language_header(raw_text, fallback_language, fallback_body))
        {
            if (!fallback_language.empty())
                detected_language = fallback_language;
            text = fallback_body;
        }
        else
        {
            text = remove_language_header(raw_text);
        }
        return 0;
    }

    std::string raw_text;
    int ret = decode(audio_embeds, "", max_new_tokens, raw_text);
    if (ret != 0)
        return ret;

    std::string detected_body;
    if (extract_language_header(raw_text, detected_language, detected_body))
    {
        if (!detected_body.empty())
        {
            text = detected_body;
            return 0;
        }

        return decode(audio_embeds, detected_language.c_str(), max_new_tokens, text);
    }

    text = remove_language_header(raw_text);
    return 0;
}

int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [wavpath] [language] [max-new-tokens=128]\n", argv[0]);
        fprintf(stderr, "       %s --batch audio_list.tsv output.csv [max-new-tokens=128]\n", argv[0]);
        return -1;
    }

    if (strcmp(argv[1], "--batch") == 0)
    {
        if (argc < 4)
        {
            fprintf(stderr, "Usage: %s --batch audio_list.tsv output.csv [max-new-tokens=128]\n", argv[0]);
            return -1;
        }

        const char* list_path = argv[2];
        const char* output_path = argv[3];
        int max_new_tokens = 128;
        if (argc >= 5)
            max_new_tokens = atoi(argv[4]);

        std::ifstream list_fp(list_path);
        if (!list_fp)
        {
            fprintf(stderr, "open %s failed\n", list_path);
            return -1;
        }

        std::ofstream out_fp(output_path);
        if (!out_fp)
        {
            fprintf(stderr, "open %s failed\n", output_path);
            return -1;
        }

        fprintf(stderr, "qwen asr: batch loading model once\n");
        std::chrono::steady_clock::time_point load_begin = std::chrono::steady_clock::now();
        Qwen3ASR asr;
        if (asr.load() != 0)
            return -1;
        std::chrono::steady_clock::time_point load_end = std::chrono::steady_clock::now();
        double load_seconds = std::chrono::duration<double>(load_end - load_begin).count();
        fprintf(stderr, "qwen asr: batch model load elapsed %.3f s\n", load_seconds);

        out_fp << "audio_relpath,ok,time_sec,language,text,error\n";

        std::string line;
        int index = 0;
        while (std::getline(list_fp, line))
        {
            if (line.empty())
                continue;

            std::string audio_path;
            std::string relpath;
            if (split_tsv_line(line, audio_path, relpath) != 0)
                continue;

            index++;
            fprintf(stderr, "qwen asr: batch [%d] %s\n", index, relpath.c_str());

            std::string text;
            std::string detected_language;
            std::string error;
            bool ok = false;

            std::chrono::steady_clock::time_point infer_begin = std::chrono::steady_clock::now();
            std::vector<short> samples;
            if (load_wav_samples(audio_path.c_str(), samples) != 0)
            {
                error = "load_wav_samples failed";
            }
            else if (asr.transcribe(samples, "", max_new_tokens, text, detected_language) != 0)
            {
                error = "transcribe failed";
            }
            else
            {
                ok = true;
            }
            std::chrono::steady_clock::time_point infer_end = std::chrono::steady_clock::now();
            double infer_seconds = std::chrono::duration<double>(infer_end - infer_begin).count();

            out_fp << csv_escape(relpath) << ','
                   << (ok ? "true" : "false") << ','
                   << infer_seconds << ','
                   << csv_escape(detected_language) << ','
                   << csv_escape(text) << ','
                   << csv_escape(error) << '\n';
            out_fp.flush();
        }

        return 0;
    }

    const char* language = "";
    int max_new_tokens = 128;
    if (argc >= 3)
    {
        if (is_decimal_number(argv[2]))
            max_new_tokens = atoi(argv[2]);
        else
            language = argv[2];
    }
    if (argc >= 4)
        max_new_tokens = atoi(argv[3]);

    std::vector<short> samples;
    if (load_wav_samples(argv[1], samples) != 0)
        return -1;

    Qwen3ASR asr;
    if (asr.load() != 0)
        return -1;

    std::string text;
    std::string detected_language;
    if (asr.transcribe(samples, language, max_new_tokens, text, detected_language) != 0)
        return -1;
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (language[0])
        fprintf(stdout, "%s\nelapsed = %.3f s\n", text.c_str(), elapsed_seconds);
    else
        fprintf(stdout, "language = %s\ntext = %s\nelapsed = %.3f s\n", detected_language.c_str(), text.c_str(), elapsed_seconds);

    return 0;
}

