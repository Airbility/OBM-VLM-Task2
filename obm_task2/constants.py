"""Fixed vocabularies and factor levels for the Task 2 dataset."""

from __future__ import annotations

# Original study vocabulary from Object-Binding-Memory-VLM.
COLORS = {
    "red": "#e84c56",
    "blue": "#3986e8",
    "green": "#37ab79",
    "yellow": "#f3be3e",
    "purple": "#a76acb",
    "orange": "#f09342",
}
SHAPES = ["circle", "square", "triangle", "hexagon", "star", "diamond"]
TEXTURES = ["dots", "diagonal_stripes", "grid", "checkerboard", "waves", "chevrons"]

# Held out of every study list. Used only for unseen features:
# unrelated probes, the novel color of near probes, and distractor attributes
# that must not belong to the study set.
EXTRA_COLORS = {
    "cyan": "#3ec6e0",
    "pink": "#e86a9a",
}
EXTRA_SHAPES = ["ellipse", "pentagon"]
EXTRA_TEXTURES = ["horizontal_stripes", "rings"]

ALL_COLORS = {**COLORS, **EXTRA_COLORS}
NEAR_COLOR = "pink"
UNRELATED = ("cyan", "ellipse", "horizontal_stripes")

ZH = {
    "color": {
        "red": "红", "blue": "蓝", "green": "绿", "yellow": "黄",
        "purple": "紫", "orange": "橙", "cyan": "青", "pink": "粉",
    },
    "shape": {
        "circle": "圆", "square": "方", "triangle": "三角", "hexagon": "六边形",
        "star": "星形", "diamond": "菱形", "ellipse": "椭圆", "pentagon": "五边形",
    },
    "texture": {
        "dots": "圆点", "diagonal_stripes": "斜条纹", "grid": "网格",
        "checkerboard": "棋盘格", "waves": "波纹", "chevrons": "折线纹",
        "horizontal_stripes": "横条纹", "rings": "圆环",
    },
}

# Study size and distractor count stay fixed so they are not factors.
STUDY_COUNT = 6
DISTRACTOR_COUNT = 3
REPETITIONS = (1, 2, 3)
OVERLAPS = (0, 1, 2)
CONTEXT_NEWLINES = (0, 1024, 2048, 4096, 8192, 16384)
SEEDS = range(1, 25)
DEV_SEEDS = range(1, 7)

INK = (0x24, 0x2A, 0x35)
BACKGROUND = (0xF6, 0xF7, 0xF9)
IMAGE_SIZE = 384

INSTRUCTION = (
    "请依次观察学习图片。之后出现的换行和干扰图片都不计入学习集合。"
    "测试图片的颜色、形状和纹理组合是否在某一张学习图片中完整出现过？"
    "位置、大小和纹理细节位置的轻微变化不改变身份。仅输出 YES 或 NO。"
)
FILLER_PREFIX = "下面是与任务无关的换行，它们不计入学习集合。"
FILLER_SUFFIX = "以上换行不计入学习集合。"
DISTRACTOR_CUE = "下面是干扰图片，它们不计入学习集合。"
TEST_CUE = "现在观察最后一张测试图片，只回答 YES 或 NO。"

DATASET_VERSION = "task2-v3-fine-grid-digit-filler"


def filler_digit_block(
    digit_count: int,
    *,
    seed: int,
    repetition: int,
    overlap: int,
) -> str:
    """Deterministic random digits for reproducible prompts."""
    import random

    if digit_count not in CONTEXT_NEWLINES:
        raise ValueError(f"digit count must be one of {CONTEXT_NEWLINES}")
    rng = random.Random(f"filler-{seed}-{repetition}-{overlap}-{digit_count}")
    return "".join(rng.choices("0123456789", k=digit_count))


def filler_text(
    digit_count: int,
    *,
    seed: int,
    repetition: int,
    overlap: int,
) -> str:
    """Filler between study and distractor cues: prefix, newline, n digits, newline, suffix."""
    if digit_count not in CONTEXT_NEWLINES:
        raise ValueError(f"digit count must be one of {CONTEXT_NEWLINES}")
    digits = filler_digit_block(
        digit_count, seed=seed, repetition=repetition, overlap=overlap
    )
    return FILLER_PREFIX + "\n" + digits + "\n" + FILLER_SUFFIX


def object_zh(color: str, shape: str, texture: str) -> str:
    return f"{ZH['color'][color]}色{ZH['shape'][shape]}形＋{ZH['texture'][texture]}"
