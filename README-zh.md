<p align="right">
  🌐 <a href="README.md">English</a> | <a href="README-zh.md">简体中文</a>
</p>

# 🎶 TIMBRAL

Toolkit for Ingestion and Model-Based Representation Archival and Loading (TIMBRAL) —
模型无关、缓存驱动的管线，把音频数据集转成可复用的 embedding。

顶层 Python 包为 `timbral`，当前包含数据集准备与 embedding 提取组件：

- **`timbral.datasets`** — 原始音频分类数据集按划分完成切片、重采样和声道转换, 构建并保存 Hugging Face `DatasetDict` 缓存。详见 [docs/datasets.md](docs/datasets.md)。
- **`timbral.embeddings`** — 从 `timbral.datasets` 生成的 raw cache 批量提取预训练音频 Encoder 的 clip/frame embedding，并保存为可复用的 Hugging Face `DatasetDict` 缓存。

## 仓库结构

```
.project-root        # rootutils 根定位标记(全仓库唯一)
pyproject.toml       # 唯一的项目/依赖/pytest 配置
assets/              # 资产目录(纯数据, 零代码), 按组件分子树
  datasets/splits/   #   各数据集划分 JSON({数据集名}/default.json, 缺失时
                     #   自动生成; 本机生成产物, 不入版本控制)
src/timbral/
  paths.py           # 仓库根定位(唯一机制), 各组件由此推导资产目录
  storage.py         # S3 存储参数、缓存目标解析与 map 临时目录管理
  datasets/          # 数据集构建组件
    adapters/        #   每数据集一个标注适配器(ADAPTERS 注册表)
    split_generators/#   每数据集一个默认划分生成器(GENERATORS 注册表, 与 adapters 同构)
  embeddings/        # embedding 提取配置、标签转换与构建编排
  models/            # 已注册的音频 Transform/Encoder
scripts/             # 命令行入口(薄调度层)
  raw_prep.py        #   构建 raw cache
  emb_prep.py        #   从 raw cache 提取 embedding
  gen_default_split.py
tests/
  datasets/          # datasets 组件测试
  embeddings/        # embeddings 组件测试
  models/            # models 组件测试
  scripts/           # 命令行入口测试
```

## 环境配置

### 依赖

要求 Python >= 3.12。运行时依赖与版本下限全部写在 [pyproject.toml](pyproject.toml),
核心是 `datasets`、`torch`/`torchaudio`、`transformers`、`librosa`/`soundfile`、
`huggingface_hub` 与 `rootutils`。其中 `datasets` 固定 5.x: 本仓库依赖其
`map(num_proc=...)` 的进程语义, 换大版本会静默改变行为。

需要特定 CUDA/ROCm 版本时, 先按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/)
装好匹配的 `torch` 与 `torchaudio`, 再安装本包, 以免被默认 wheel 覆盖。

### 安装

```bash
git clone <this-repo> && cd timbral
python -m venv .venv && source .venv/bin/activate   # 或 conda create -n timbral python=3.12
pip install -e .
```

安装本包时用可编辑安装: 导入期由 rootutils 查找仓库根的 `.project-root` 标记, 而该标记
不随包分发, 普通安装后根定位会失败。

`scripts/*.py` 会自行定位仓库根并注入 `src` 导入路径, 因此无需设置 `PYTHONPATH`, 只在仓库外
`import timbral` 时才需要上面的可编辑安装。

BEATs 权重下载脚本另有一组可选依赖(它独立于运行时, 不参与管线):

```bash
pip install -e ".[beats-dl]" && playwright install chromium
```

### 环境变量

全部可选, 按需设置:

| 变量 | 作用 | 默认 |
|---|---|---|
| `HF_HOME` / `HF_HUB_CACHE` | Hugging Face 缓存根; 本仓库的编码器权重落在 `$HF_HUB_CACHE/audioencoders/` 下(PANNs 按模型名、AST/CLAP/wav2vec2 按 repo_id、BEATs 统一在 `beats/`) | `~/.cache/huggingface`, `$HF_HOME/hub` |
| `TMPDIR` | 输出到 `s3://` 时 map 临时 Arrow 文件的落盘位置 | 系统临时目录 |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_S3_ENDPOINT` | 输出到 `s3://` 时的凭据与 endpoint(非 AWS 的兼容对象存储必须给 endpoint) | 无 |

## 使用教程

管线分两段: `raw_prep.py` 把原始音频按划分切片、重采样、转声道, 落成 raw cache;
`emb_prep.py` 从 raw cache 批量前向预训练 Encoder, 落成 embedding cache。两段都以配置
哈希命名输出目录, 同参重跑天然命中已有产物并跳过。

一条端到端的最小链路:

```bash
python scripts/raw_prep.py \
    --dataset_name ESC-50 --dataset_dir /path/to/ESC-50 \
    --cache_dir /path/to/raw_cache/ESC-50

python scripts/emb_prep.py \
    --cache_dir /path/to/raw_cache/ESC-50 \
    --model_name panns-32k-cnn14-max_mean \
    --granularity clip \
    --output_dir /path/to/emb_cache
```

### 1. 准备数据集源文件

数据集本身需要自行从官方渠道获取并解压, `--dataset_dir` 指向解压后的数据集根目录;
各适配器按官方发布的目录结构与标注文件解析, 具体预期见
[src/timbral/datasets/adapters/](src/timbral/datasets/adapters/) 中同名模块。

已注册适配器的数据集(`--dataset_name` 的取值):

`AudioSetStrong`、`AudioSetWeak`、`BSD10K`、`BSD35K`、`BirdVox-14SD`、`DB3V`、
`DCASE-2024-Task-5`、`DESED`、`DataSED`、`ESC-50`、`FSD50K`、`FSDnoisy18k`、
`HyenaSET`、`RealDESED`、`SONYC-UST`、`UrbanSound8K`

### 2. 构建 raw cache

```bash
python scripts/raw_prep.py \
    --dataset_name ESC-50 \
    --dataset_dir /path/to/ESC-50 \
    --cache_dir /path/to/raw_cache/ESC-50 \
    --sr 32000 --seg_sec 10.0 --hop_sec 10.0 \
    --num_proc 4 --batch_size 16
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset_name` | 必填 | 数据集名称, 取值见上表 |
| `--dataset_dir` | 必填 | 数据集源文件根目录 |
| `--cache_dir` | `{cwd}/{dataset_name}/{config_hash}` | 输出目录, 支持本地路径与 `s3://` |
| `--split_json` | `assets/datasets/splits/{dataset_name}/default.json` | 划分文件; 默认文件缺失时自动生成, 显式路径缺失时报错 |
| `--sr` | `16000` | 目标采样率, 建议按下游模型的原生采样率设置 |
| `--mono` / `--no-mono` | `--mono` | 是否混为单声道 |
| `--seg_sec` / `--hop_sec` | `10.0` / `10.0` | 切片长度与步长(秒) |
| `--tol_sec` | `0.0` | 尾部短切片的最小保留长度, 仅作用于 `segment_id > 0` |
| `--label_type` | `weak` | `weak` 逐切片聚合标签, `strong` 保留切片内事件列表 |
| `--num_proc` / `--batch_size` | `4` / `16` | map 的进程数与批大小; `--num_proc 0` 表示在主进程内处理(datasets 5.x 中 `1` 反而会开一个工作进程) |
| `--overwrite` | 关 | 目录已存在时强制重建(默认跳过, 跳过前校验 `config_hash` 一致) |

`config_hash` 由 `dataset_name`、划分文件哈希、`sr`、`mono`、`seg_sec`、`hop_sec`、
`tol_sec`、`label_type` 共同计算, 因此改参必然换目录。输出目录内除 `DatasetDict` 外还有
`prep_config.json` 与 `label_index.json`(类名到 index 的映射)。

划分文件默认取 `assets/datasets/splits/{dataset_name}/default.json`, 不存在时由注册的
生成器现场生成。也可以单独生成或生成到指定位置:

```bash
python scripts/gen_default_split.py \
    --dataset_name ESC-50 --dataset_dir /path/to/ESC-50 [--output /path/to/split.json]
```

划分 JSON 的格式约定、k-fold 数据集的组织方式, 以及变长(ragged)数据集的切片取舍, 见
[docs/datasets.md](docs/datasets.md)。

### 3. 提取 embedding

```bash
python scripts/emb_prep.py \
    --cache_dir /path/to/raw_cache/ESC-50 \
    --model_name panns-32k-cnn14-max_mean \
    --granularity clip \
    --output_dir /path/to/emb_cache \
    --device auto --batch_size 32
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cache_dir` | 必填 | `raw_prep.py` 产出的缓存目录(本地路径) |
| `--model_name` | 必填 | 已注册的模型名, 见下表 |
| `--granularity` | 必填 | `clip`(整段一个 `[D]` 向量)或 `frame`(逐帧 `[T, D]`) |
| `--output_dir` | 当前工作目录 | 输出根目录前缀, 支持本地路径与 `s3://` |
| `--device` | `auto` | `auto` 按 cuda > mps > cpu 选择, 也可写 `cpu`/`cuda:0`/`mps` |
| `--batch_size` | `32` | 每次前向的切片数 |
| `--pretrained_dir` | 无 | 自定义权重目录, 默认用 Hugging Face 缓存下的项目专属目录 |
| `--overwrite` | 关 | 输出已存在时删除重建 |

输出路径恒为三级结构 `{output_dir}/{dataset_name}/{model_name}/{emb_hash}`, 模型名中的
`/` 替换为 `--`, `emb_hash` 由 raw cache 的 `config_hash`、`model_name` 与 `granularity`
计算(设备、批大小等执行参数不参与)。目录内含 `DatasetDict`、从 raw cache 拷贝的
`label_index.json`, 以及最后写入的 `emb_config.json`(完整参数快照, 兼作完成标记)。

数据形态由缓存自动推断, 无需额外声明: weak 缓存标签直通; strong + clip 逐类聚合为 `[C]`
三态 multi-hot; strong + frame 生成 `[T, C]` 帧标签。frame 粒度另外落盘时间几何
`geometry [T, 2]` 与有效帧掩码 `valid_mask [T]`, 无效帧的 embedding/label/geometry 一律
零填充, 有效性以 `valid_mask` 为准。

已注册的 `--model_name` 取值:

| 家族 | 注册名 |
|---|---|
| PANNs | `panns-16k-cnn14-max_mean`、`panns-32k-cnn14-max_mean`、`panns-32k-cnn14-decision_level_max` |
| AST | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| CLAP | `laion/clap-htsat-fused`(仅 clip 粒度) |
| wav2vec2 | `facebook/wav2vec2-base` |
| BEATs | `beats_iter1`/`beats_iter2`/`beats_iter3`/`beats_iter3_plus_as20k`/`beats_iter3_plus_as2m`, 及对应的 `fine_tuned_*_cpt1`/`cpt2`, 共 15 个 |

程序内取全量列表: `from timbral.models import list_models; list_models()`。

### 4. 预训练权重

PANNs、AST、CLAP、wav2vec2 的权重在首次构造模型时自动下载到
`$HF_HUB_CACHE/audioencoders/` 下, 并按固定 SHA-256 校验, 无需手工准备。

BEATs 官方只通过 OneDrive 分享链接发布权重, 运行时不含下载代码, 需要先跑独立脚本
(依赖 playwright, 见上文可选依赖):

```bash
python scripts/extra/beats_dl.py \
    --dest ~/.cache/huggingface/hub/audioencoders/beats \
    --entries beats_iter3_plus_as2m fine_tuned_beats_iter3_cpt1 \
    --workers 3
```

`--dest` 给默认权重目录(`$HF_HUB_CACHE/audioencoders/beats`)时, `emb_prep.py` 无需额外
参数即可命中; 放在别处则用 `--pretrained_dir` 指过去。`--entries` 省略时下载全部 15 个
checkpoint。机制详见
[docs/designs/models/extra/beats-download.md](docs/designs/models/extra/beats-download.md)。

### 5. 读取产物

两段缓存都是标准的 Hugging Face `DatasetDict`, 脱离本仓库也能读:

```python
from datasets import load_from_disk

emb = load_from_disk("/path/to/emb_cache/ESC-50/panns-32k-cnn14-max_mean/<emb_hash>")
train = emb["train"].with_format("numpy")
print(train.column_names)
# ['audio_path', 'audio_id', 'segment_id', 'start', 'end', 'valid_sec',
#  'embedding', 'label']  (frame 粒度另有 'geometry' 与 'valid_mask')
print(train[0]["embedding"].shape)
```

`with_format("numpy")` 比 `"torch"` 明显更快, 建议在 DataLoader 侧再转张量。类名与 index
的对应关系读同目录的 `label_index.json`。

### 6. 输出到对象存储

`--cache_dir` / `--output_dir` 传 `s3://bucket/prefix` 时, map 临时 Arrow 写入 `$TMPDIR`,
最终产物直接写入对象存储。使用前配好凭据与 `AWS_S3_ENDPOINT`(见环境变量表)。S3 路径必须
包含桶内目录, 不能用桶根。

## 引用

如果 TIMBRAL 对您的研究有所帮助，请考虑引用本仓库：

```bibtex
@misc{zhang2026timbral,
  author       = {Shiqi Zhang},
  title        = {{TIMBRAL}: Toolkit for Ingestion and Model-Based Representation Archival and Loading},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/TioSisai/timbral}
}
```
