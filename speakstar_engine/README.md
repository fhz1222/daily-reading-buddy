# Speak Star 本地音素引擎

按 [Local Pronunciation Engine Plan v2](../docs/pronunciation-engine-plan.md) 实现的研究原型，默认英式 `en-gb`。提供音素对齐、GOP、替换/删除证据，以及校准与独立评测工具。不评重音、语调、节奏，不把目标句冒充识别结果。

**无校准文件时：所有音素 `uncertain`，分数为 `null`。** 有校准文件不等于已经验证；当前输出始终带 `engine.validated: false`，不接入孩子页面进行自动纠错。原有 Whisper 页面继续工作。

## 安装（Windows，项目根目录）

```powershell
python -m venv .venv-engine
.venv-engine/Scripts/python.exe -m pip install -e ./speakstar_engine
```

运行依赖只装在这个环境中。eSpeak NG 的库及数据通过 `espeakng-loader` 提供，ffmpeg 可使用本机版本或包内版本，不需要修改系统 PATH。

首次导出模型需要联网下载约 1.2 GB 权重，以及 PyTorch 等开发依赖：

```powershell
.venv-engine/Scripts/python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv-engine/Scripts/python.exe -m pip install -e "./speakstar_engine[export]"
.venv-engine/Scripts/python.exe speakstar_engine/scripts/export_onnx.py --output models/wav2vec2
```

导出会生成 fp32 与实验性 int8，检查两种音频长度的输出，并记录大小、加载时间、推理耗时和数值差异。默认使用 fp32；int8 必须单独提取特征、校准并通过准确率比较，不能复用 fp32 的校准文件。

已有缓存时加 `--local-files-only`；日常评测只读取本地模型，不发起网络调用。导出后不需要 PyTorch 或 Transformers。

## 评测音频

```powershell
.venv-engine/Scripts/python.exe -m speakstar clip.wav "Three thin thumbs." --locale en-gb --evidence --output artifacts/result.json
```

支持 WAV、WebM、MP3、Ogg、MP4，最长 12 秒；数字请写成英文。`--evidence` 用于研究和校准，返回的是模型证据，不是发音正确率。初始静音被裁剪后，时间偏移仍对应原录音。低能量检测只能筛掉明显静音，不能保证识别出噪声。

`NoSpeech` 表示过短或明显静音；`UncertainAlignment` 表示无法对齐。终端错误以 JSON 写到 stderr，并返回退出码 2。

## 校准与评测

下载并解压 [speechocean762](https://www.openslr.org/101)，保留 `scores.json`、`train/`、`test/`、`WAVE/` 的目录结构。以下路径示例用 `datasets/speechocean762`；原始公开数据无需与代码一起提交。

```powershell
.venv-engine/Scripts/python.exe speakstar_engine/scripts/split_so762.py datasets/speechocean762 --output artifacts/splits.json
.venv-engine/Scripts/python.exe speakstar_engine/scripts/extract_so762.py datasets/speechocean762/scores.json artifacts/splits.json --splits train dev --output artifacts/train-dev.jsonl
.venv-engine/Scripts/python.exe speakstar_engine/scripts/calibrate.py artifacts/train-dev.jsonl --output artifacts/calibration.fp32.en-gb.json
```

注意：

- 按说话者拆分训练和开发数据，保留官方测试集。测试数据不能送入校准脚本。
- ARPAbet 与 eSpeak IPA 不总是一一对应，尤其英美口音及弱读。提取器仅在整词音素一致时关联标签，跳过情况写入 `.coverage.json`；不得隐藏这些跳过后宣称覆盖所有发音。
- 专家平均分不是离散的 0/1/2。当前标签策略明确固定为：平均分 ≥1 为正确（允许口音），≤0.4 为错误，其间暂不用于二分类。需要一并报告被排除的标签。
- 全局 logistic 校准只产生初步概率；某音素在开发集至少有 20 个正确、10 个错误样本，开发 ECE ≤0.1 且时间跨度合理时，才可给出高置信判断。
- 替换提示和漏音提示默认禁用：当前自动校准尚未根据独立的替换/遗漏标签选择这两个阈值。原始证据可查看，但不能冒充已验证的具体纠错。
- `--limit` 只用于流程冒烟，不构成准确率评估。完整提取计算量较大。

**冻结阈值、验收标准后再读取测试音频：**

```powershell
.venv-engine/Scripts/python.exe speakstar_engine/scripts/extract_so762.py datasets/speechocean762/scores.json artifacts/splits.json --splits test --output artifacts/test.jsonl
.venv-engine/Scripts/python.exe speakstar_engine/scripts/evaluate.py artifacts/test.jsonl --calibration artifacts/calibration.fp32.en-gb.json --output artifacts/test-report.json
.venv-engine/Scripts/python.exe -m speakstar clip.wav "Three thin thumbs." --calibration artifacts/calibration.fp32.en-gb.json
```

报告包括儿童子集、误报率、错误检测 precision/recall/F1、不确定比例、ECE、音素分相关性及有标签替换提示准确率；没有足够数据时返回 `null`。不确定结果计入漏检，防止通过全部弃权获得好指标。

报告不会自动批准模型。还需审查提取覆盖率、独立词级相关性、公开基线比较、替换和遗漏阈值、fp32/int8 差异，再决定是否通过 v2 里程碑 4。家长标注适合个人试用验证，不等同于独立专家验证。

## 测试

```powershell
.venv-engine/Scripts/python.exe -m unittest discover -s speakstar_engine/tests -v
.venv-engine/Scripts/python.exe -m unittest discover -s tests -v
```

包含 CTC 全路径穷举核对、与 torchaudio 对齐比较、重复音、真实 eSpeak G2P、静音、时间偏移、置信门控、跨模型校准拒绝，以及数据泄漏检查。没有安装 torchaudio 时，其参考比较会明确跳过。

## 当前交付边界

已实现可运行的模型导出、音素证据 CLI、校准/评测工具链。**尚未完成**完整公开数据集校准、替换/遗漏反馈阈值调优、儿童独立准确率验收、网页集成及个人调音。通过验收前不会将原型称为可靠发音评测工具。

模型权重、环境和实验结果位于忽略目录 `models/`、`.venv-engine/`、`artifacts/`。录音只由解码进程读取，不被引擎另存或上传；CLI 输出文件仅在明确指定 `--output` 时保存。
