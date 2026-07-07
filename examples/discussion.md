# Qwen3-ASR 0.6B / 1.7B 在 ncnn 上的推理部署

**本文介绍如何将 Qwen3-ASR 0.6B 与 Qwen3-ASR 1.7B 部署到 ncnn，并使用 C++ 示例程序完成离线语音识别。项目目前提供两个可执行示例：**

- `qwen_asr_0_6_b.cpp`：Qwen3-ASR-0.6B 推理程序
- `qwen_asr_1_7_b.cpp`：Qwen3-ASR-1.7B 推理程序

**两份代码均支持：**

- 单条 wav 音频识别
- 批量音频识别并导出 CSV
- 指定识别语言，减少模型自行预测语言的 token 开销
- 限制最大输出 token 数
- 音频侧 Vulkan 加速
- 文本侧 Vulkan 实验开关

## 项目目标

### 流程概览

Qwen3-ASR 是一个基于多模态大模型结构的语音识别模型。它不是传统的单一 encoder-decoder ASR 网络，而是由音频编码模块和文本生成模块共同完成识别：

1. 音频输入先被转换为 fbank 特征。
2. 音频特征经过 audio CNN、audio transformer 和 audio projection 得到音频 embedding。
3. C++ 程序构造 Qwen chat prompt，并把音频 embedding 填入 `<audio_pad>` 对应位置。
4. 文本 decoder 根据 prompt 和音频 embedding 自回归生成识别文本。

**本项目的目标是将上述流程尽量完整地迁移到 ncnn C++ 推理环境中，使其可以在本地离线运行，并适配 CPU / Vulkan 两种推理路径。**

## 模型导出

### 导出内容

Qwen3-ASR 的模型可以拆分为以下模块：

- 音频 encoder：`audio_cnn`、`audio_transformer`、`audio_proj`
- 文本 embedding：`text_embed`
- 文本 decoder：`text_decoder_prefill_layer_XX`、`text_decoder_step_layer_XX`
- 文本输出：`text_norm`、`lm_head`
- 前处理资源：`mel_filters.f32.bin`、`vocab.json`、`merges.txt`

**模型的大部分静态图可以使用以下项目导出：**

```text
https://github.com/Chisato623/QWEN-ASR-PNNX
```

**也可以直接下载已经导出的 ncnn 模型文件：**

```text
https://github.com/Chisato623/QWEN-ASR-2-ncnn/releases
```

**需要注意的是，Qwen3-ASR 中仍有一部分动态逻辑不适合直接导出为单个静态计算图，例如 prompt 构造、语言控制、KV cache 更新、自回归 token 选择等。因此这些逻辑在 C++ 示例程序中手动实现。**

## 音频处理

### 输入约束

本项目参考了 ncnn 官方示例中的 Whisper 实现：

```text
https://github.com/Tencent/ncnn/blob/master/examples/whisper.cpp
```

**当前程序只支持以下 wav 输入格式：**

- PCM s16le
- 16 kHz 采样率
- 单声道
- wav 容器

**相比 Whisper 示例中常见的 15 秒音频输入，本项目将音频最大处理长度扩展到约 30 秒。程序内部最多读取 `480000` 个采样点，对应：**

```text
16000 samples/s * 30 s = 480000 samples
```

### 处理流程

1. 读取 wav 采样点。
2. 将 `int16` 音频归一化为 float。
3. 使用 ncnn `Spectrogram` layer 计算频谱。
4. 使用 `mel_filters.f32.bin` 转换为 128 维 fbank。
5. 对 fbank 做 log、clamp 和归一化。
6. 将 3000 帧特征切成 30 个 chunk 输入音频 encoder。

## 文本解码

### 解码方式

**Qwen3-ASR 的文本生成分为两步：**

1. `prefill`：一次性处理完整 prompt，得到第一步 logits 和每层 KV cache。
2. `step`：每次输入一个 token，更新 KV cache，并生成下一个 token。

**程序中使用 greedy search：**

```cpp
next_id = argmax(logits)
```

**当生成到 `<|im_end|>` 或 `<|endoftext|>` 时停止。指定语言时，程序会把类似下面的文本提前放入 prompt：**

```text
language Chinese<asr_text>
```

**这样模型可以跳过语言预测部分，直接生成识别文本。通常这会减少 decoder step 次数，因此比不指定语言略快。**

## 目录结构

### 文件布局

**模型文件需要放在对应目录中：**

```text
examples/
  qwen_asr_0_6_b.cpp
  qwen_asr_1_7_b.cpp
  Qwen3-ASR-0.6B/
    qwen3_asr_0_6b_audio_cnn.ncnn.param
    qwen3_asr_0_6b_audio_cnn.ncnn.bin
    qwen3_asr_0_6b_audio_transformer.ncnn.param
    qwen3_asr_0_6b_audio_transformer.ncnn.bin
    qwen3_asr_0_6b_audio_proj.ncnn.param
    qwen3_asr_0_6b_audio_proj.ncnn.bin
    qwen3_asr_0_6b_text_embed.ncnn.param
    qwen3_asr_0_6b_text_embed.ncnn.bin
    qwen3_asr_0_6b_text_decoder_prefill_layer_00..27.ncnn.param/bin
    qwen3_asr_0_6b_text_decoder_step_layer_00..27.ncnn.param/bin
    qwen3_asr_0_6b_text_norm.ncnn.param
    qwen3_asr_0_6b_text_norm.ncnn.bin
    qwen3_asr_0_6b_lm_head.ncnn.param
    qwen3_asr_0_6b_lm_head.ncnn.bin
    mel_filters.f32.bin
    vocab.json
    merges.txt
  Qwen3-ASR-1.7B/
    qwen3_asr_1_7b_*.ncnn.param/bin
    mel_filters.f32.bin
    vocab.json
    merges.txt
```

**程序默认从当前工作目录加载模型文件。因此建议进入模型目录后运行可执行文件。**

## 编译方法

### 构建命令

在 ncnn 根目录下使用 CMake 构建：

```powershell
cmake --build build --target qwen_asr_0_6_b -j 8
cmake --build build --target qwen_asr_1_7_b -j 8
```

**编译完成后，可执行文件位于：**

```text
build/examples/qwen_asr_0_6_b.exe
build/examples/qwen_asr_1_7_b.exe
```

### Vulkan 配置

**如果需要 Vulkan 支持，ncnn 需要在配置阶段开启：**

```text
NCNN_VULKAN=ON
```

## 单文件识别

### Qwen3-ASR-0.6B

进入模型目录：

```powershell
cd E:\ncnn\examples\Qwen3-ASR-0.6B
```

**自动检测语言：**

```powershell
..\..\build\examples\qwen_asr_0_6_b.exe input.wav
```

**指定语言：**

```powershell
..\..\build\examples\qwen_asr_0_6_b.exe input.wav Chinese
..\..\build\examples\qwen_asr_0_6_b.exe input.wav English
```

**指定最大输出 token 数：**

```powershell
..\..\build\examples\qwen_asr_0_6_b.exe input.wav Chinese 128
```

### Qwen3-ASR-1.7B

进入模型目录：

```powershell
cd E:\ncnn\examples\Qwen3-ASR-1.7B
```

**运行：**

```powershell
..\..\build\examples\qwen_asr_1_7_b.exe input.wav Chinese 128
```

## 批量识别

### 命令

**批处理模式会只加载一次模型，然后连续识别多个音频文件，适合数据集测试。**

命令格式：

```powershell
qwen_asr_0_6_b.exe --batch audio_list.tsv output.csv [max-new-tokens]
qwen_asr_1_7_b.exe --batch audio_list.tsv output.csv [max-new-tokens]
```

### TSV 格式

**`audio_list.tsv` 每行包含两列，用 tab 分隔：**

```text
音频路径	输出中的相对路径
```

**示例：**

```text
E:\dataset\audio_0001.wav	audio_0001.wav
E:\dataset\audio_0002.wav	audio_0002.wav
```

**运行示例：**

```powershell
cd E:\ncnn\examples\Qwen3-ASR-0.6B
..\..\build\examples\qwen_asr_0_6_b.exe --batch list.tsv result.csv 128
```

### CSV 输出

**输出 CSV 字段为：**

```text
audio_relpath,ok,time_sec,language,text,error
```

## Vulkan 加速

**程序默认关闭 Vulkan，因为部分文本侧网络在 Vulkan 下可能出现精度差异，导致生成文本偏离 CPU 结果。**

### 音频侧

**启用音频侧 Vulkan：**

```powershell
$env:NCNN_QWEN_ASR_GPU = "0"
```

**默认情况下，开启该变量后，以下模块会使用 Vulkan：**

- `audio_cnn`
- `audio_transformer`
- `audio_proj`

**文本 decoder 默认仍使用 CPU，以保证识别文本稳定。**

### Vulkan fp16

**默认 Vulkan 使用 fp32，以尽量贴近 CPU 输出。如果想测试 fp16：**

```powershell
$env:NCNN_QWEN_ASR_FP16 = "1"
```

**fp16 可以降低显存占用，但可能放大自回归文本生成中的数值误差。**

### 文本侧 Vulkan 实验开关

**文本侧 Vulkan 目前作为实验功能保留：**

```powershell
$env:NCNN_QWEN_ASR_TEXT_GPU = "1"
```

**也可以只打开部分模块：**

```powershell
$env:NCNN_QWEN_ASR_TEXT_EMBED_GPU = "1"
$env:NCNN_QWEN_ASR_TEXT_HEAD_GPU = "1"
$env:NCNN_QWEN_ASR_PREFILL_GPU_LAYERS = "4"
$env:NCNN_QWEN_ASR_STEP_GPU_LAYERS = "4"
```

**在测试中，全量文本 fp32 Vulkan 可能出现显存分配失败，全量文本 fp16 Vulkan 可能导致输出文本重复或错误。因此推荐默认配置为：**

```text
音频侧 Vulkan + 文本侧 CPU
```

## 测试结果示例

**在 Qwen3-ASR-0.6B 上，使用以下测试音频：**

```text
E:\QWEN-ASR2PNNX\asr_zh.wav
E:\QWEN-ASR2PNNX\asr_en.wav
```

**CPU baseline 输出：**

```text
asr_zh.wav: 甚至出现交易几乎停滞的情况。
asr_en.wav: Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
```

**开启音频侧 Vulkan 后，两条音频的识别文本与 CPU baseline 一致，同时推理时间明显降低。**

## 当前限制

- 只支持 16 kHz、单声道、PCM s16le wav。
- wav 解析较严格，不完整支持带复杂 metadata chunk 的 wav 文件。
- 单段音频最长约 30 秒，超过部分会被截断。
- 文本 decoder 采用 greedy search，没有实现 beam search、temperature、top-k 或 top-p。
- 文本侧 Vulkan 仍属于实验功能，可能受显存、精度和 CPU/GPU 同步开销影响。
- 当前实现为了方便部署，将多个子图拆成独立 ncnn 模型，加载文件数量较多。

## 总结

本项目完成了 Qwen3-ASR 0.6B 和 1.7B 在 ncnn C++ 环境中的基本部署，覆盖了音频前处理、音频 encoder、Qwen prompt 构造、文本 prefill、KV cache 更新、step 解码、语言控制和批处理评测等完整流程。

**目前推荐的使用方式是：**

```text
CPU 文本解码保证正确性，Vulkan 加速音频 encoder。
```

这种配置在保持识别结果稳定的同时，可以显著降低整体推理耗时，是当前工程实现中较稳妥的折中方案。
