# OBM-VLM Task 2

比较三个因素如何影响视觉语言模型的物体绑定：上下文距离、特征重叠、特征重复。学习图和测试图是 384×384 的单个居中物体，由颜色、形状、纹理定义。测试图相对学习图有轻微平移和缩放。

数据版本是 `task2-v3-fine-grid-digit-filler`。正式比较只用 test（种子 7–24）。YES 只占每组 8 题中的 3 题，不要用总准确率当主指标。

## 目录

```text
Dataset.tar.gz            题目、图片、标签的压缩包。使用前解压出 Dataset/
dataset_api.py            读取题目，不把答案混进模型输入
build_dataset.py          重新生成 Dataset/
validate_dataset.py       检查因素是否分开
obm_task2/                数据集生成与校验
run_eval_qwen3vl.py       Qwen3-VL
run_eval_kimi_vl.py       Kimi-VL-A3B
run_eval_llava_next.py    LLaVA-NeXT
run_eval_deepseek_flash.py  DeepSeek-V4.1-Flash（思考关闭）
analyze_results.py        由 predictions.jsonl 画图并写 REPORT.md
results/                  已跑完的 test 结果
requirements.txt          生成数据集所需依赖
```

## 数据集

10368 题，975 张图。24 个种子：1–6 是开发集，7–24 是测试集。学习图始终 6 张，干扰图始终 3 张。三个因素全交叉，每个条件 8 题：3 道 target（YES）、3 道 critical recombination（NO）、1 道 near lure（NO）、1 道 unrelated lure（NO）。

| 因素 | 取值 | 改了什么 |
|---|---|---|
| `repetition` | 1、2、3 | 1：六种属性各出现 1 次。2：三种属性各出现 2 次。3：三种属性按 3、2、1 次出现 |
| `overlap` | 0、1、2 | 每张干扰图与学习集合共享 0、1 或 2 个属性 |
| `context_newlines` | 0、1024、2048、4096、8192、16384 | 学习图和干扰图之间的随机数字个数。字段名沿用旧称，内容是数字 |

题目顺序固定：说明 → 6 张学习图 → 数字间隔 → 「下面是干扰图，不计入学习集合」→ 3 张干扰图 → 「只回答 YES 或 NO」→ 测试图。拉长距离时，被推远的是学习图。干扰图和测试图挨在一起。

| 题型 | 正确答案 | 含义 |
|---|---|---|
| `target` | YES | 测试图就是某一张学习图 |
| `critical_recombination` | NO | 三个属性都学过，但这个组合以及任意两两配对都没在同一张学习图上出现过 |
| `near_lure` | NO | 颜色是没学过的粉色，形状和纹理学过，形状–纹理配对没学过 |
| `unrelated_lure` | NO | 青、椭圆、横条纹，三个属性都不在学习列表里 |

target 下降表示一般性遗忘。target 还稳、critical 的错误接受率上升，才表示绑定错误增加。

```text
Dataset/
  images/            PNG。文件名是哈希，不含颜色、形状或纹理
  prompts/           每题一个 JSON，只有文本和相对图片路径
  labels/labels.jsonl  标准答案和属性。不能发给模型
  manifest.jsonl     按条件筛选题目，不含答案
  metadata.json      规模和因素定义
```

同一 `seed`、`repetition`、`probe_index` 在不同重叠和不同数字间隔下，学习图和测试图是同一批文件。重复次数会更换学习列表，不能按 `probe_index` 配成同一道题。

## 使用数据集

数据集以 `Dataset.tar.gz` 提供。在项目根目录解压后再读取或评测：

```bash
tar -xzf Dataset.tar.gz
```

需要 Python 3.11 以上。

```bash
pip install -r requirements.txt
python validate_dataset.py
```

重新生成整个 `Dataset/`：

```bash
python build_dataset.py
```

生成结束会自己跑一遍校验。

读取题目时只把 `qwen_content` 或 `model_messages` 的文本和图片发给模型。不要发送 `trial_id`、`labels/`、`manifest.jsonl` 或属性字段。每题使用一个新的上下文。

```python
from dataset_api import Task2Dataset, parse_yes_no

dataset = Task2Dataset("Dataset")
for example in dataset.iter_examples(split="test", repetition=1, overlap=0, context_newlines=0):
    content = dataset.qwen_content(example["trial_id"])
    raw = model.generate(content)
    prediction = parse_yes_no(raw)
    label = dataset.get_label(example["trial_id"])  # 评分时才读取
```

`qwen_content` 的图片字段是 `image`，值为绝对路径。`model_messages` 的图片字段是 `path`。`parse_yes_no` 只接受去掉空白后正好是 `YES` 或 `NO` 的输出。

比较一个因素时，固定另外两个：

- 上下文距离：固定 `repetition` 和 `overlap`
- 特征重叠：固定 `repetition` 和 `context_newlines`
- 特征重复：固定 `overlap` 和 `context_newlines`，按题型汇总

## 已有结果

四个模型用的是同一套 test，每个 7776 题。发布包里是合并后的 `predictions.jsonl`、`summary.json`、报告和图，不含运行日志和分片中间文件。

| 目录 | 模型 |
|---|---|
| `results/qwen3vl-4b` | Qwen3-VL-4B-Instruct |
| `results/qwen3vl-8b` | Qwen3-VL-8B-Instruct |
| `results/kimi-vl-a3b` | Kimi-VL-A3B-Instruct |
| `results/deepseek-v4.1-flash` | DeepSeek-V4.1-Flash，思考关闭 |
| `results/model_comparison` | 四个模型的对照图和文字报告 |

`summary.json` 里的 `rate`：target 是命中率，另外三类是错误接受率（答成 YES 的比例）。`predictions.jsonl` 每一行有 `gold`、`raw`、`prediction`。

由预测重新画图：

```bash
python analyze_results.py \
  --predictions results/qwen3vl-8b/predictions.jsonl \
  --output-dir results/qwen3vl-8b/figures \
  --title Qwen3-VL-8B-Instruct
```

图会写到 `--output-dir`，`REPORT.md` 写到它的上一级。这一步需要 `numpy` 和 `matplotlib`。

## 运行评测

数据集生成只需要 `requirements.txt` 里的 Pillow。跑模型还需要对应环境里的 `torch`、`transformers`、`tqdm`；DeepSeek 脚本还需要 `openai`。权重自行准备，用本地目录传给 `--model`。

四个脚本都支持 `--split dev|test|all` 和 `--limit N`。`--split test` 是正式结果。已写入的 `trial_id` 会跳过，中断后用同一 `--output-dir` 再跑即可续上。

多卡分片时，每个进程设置自己的 `CUDA_VISIBLE_DEVICES`，并加上 `--num-shards` 和 `--shard-id`。全部分片结束后加 `--merge-only`，合并成 `predictions.jsonl` 并写 `summary.json`。

### Qwen3-VL

```bash
python run_eval_qwen3vl.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --split test \
  --output-dir results/qwen3vl-8b
```

`--device-map single`（默认）把整份模型放到当前可见的第一张卡。`--device-map auto` 按层切到所有可见 GPU，不要和分片同时用。

两张卡各跑一半：

```bash
CUDA_VISIBLE_DEVICES=0 python run_eval_qwen3vl.py --model /path/to/Qwen3-VL-8B-Instruct \
  --split test --output-dir results/qwen3vl-8b --num-shards 2 --shard-id 0
CUDA_VISIBLE_DEVICES=1 python run_eval_qwen3vl.py --model /path/to/Qwen3-VL-8B-Instruct \
  --split test --output-dir results/qwen3vl-8b --num-shards 2 --shard-id 1
python run_eval_qwen3vl.py --model /path/to/Qwen3-VL-8B-Instruct \
  --split test --output-dir results/qwen3vl-8b --num-shards 2 --merge-only
```

4B 与 8B 用同一个脚本，换 `--model` 和 `--output-dir`。

### Kimi-VL-A3B

```bash
python run_eval_kimi_vl.py \
  --model /path/to/Kimi-VL-A3B-Instruct \
  --split test \
  --output-dir results/kimi-vl-a3b
```

`--device-map` 与 Qwen 相同。`--fail-on-error` 会在单题失败时退出。分片合并同样使用 `--num-shards` 和 `--merge-only`。

### LLaVA-NeXT

```bash
python run_eval_llava_next.py \
  --model /path/to/llava-v1.6 \
  --split test \
  --output-dir results/llava-next \
  --device-map auto
```

长上下文在单卡上容易放不下。`--device-map auto` 会把一层模型均衡切到当前可见的 GPU 上，至少要两张可见卡。分片和合并参数与上面两个脚本相同。这次发布的 `results/` 里没有 LLaVA 的完成结果。

### DeepSeek-V4.1-Flash

密钥只从环境变量读取。不要把 key 写进脚本或仓库。

```bash
export DEEPSEEK_API_KEY=你的密钥
python run_eval_deepseek_flash.py --split dev --limit 8
python run_eval_deepseek_flash.py --split test --output-dir results/deepseek-v4.1-flash
```

脚本请求 `deepseek-flash`，并关闭思考（`thinking.type=disabled`）。默认 8 个并发，可用 `--workers` 调整。未设置 `DEEPSEEK_API_KEY` 时脚本会直接退出。
