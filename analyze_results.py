"""Summarize and visualize Task 2 eval results (protocol-aligned)."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# CJK labels; avoid DejaVu fallback on mixed EN/ZH strings (legend tofu).
_CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "AR PL UKai CN",
    "Droid Sans Fallback",
    "SimHei",
)
CJK_FONT: font_manager.FontProperties | None = None


def configure_matplotlib_cjk() -> font_manager.FontProperties | None:
    global CJK_FONT
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_FONT_CANDIDATES if name in available), None)
    if chosen:
        # Do not addfont() the whole TTC (first face is JP). Use family name lookup.
        plt.rcParams["font.family"] = chosen
        plt.rcParams["font.sans-serif"] = [chosen]
        CJK_FONT = font_manager.FontProperties(family=chosen)
    plt.rcParams["axes.unicode_minus"] = False
    return CJK_FONT


def _fp() -> dict:
    if CJK_FONT is None:
        return {}
    return {"fontproperties": CJK_FONT}


def _set_cjk_xticklabels(ax, labels, **kwargs) -> None:
    ax.set_xticklabels(labels, **kwargs)
    if CJK_FONT is not None:
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(CJK_FONT)


PROBE_TYPES = ("target", "critical_recombination", "near_lure", "unrelated_lure")
PROBE_LABELS = {
    "target": "Target 命中率",
    "critical_recombination": "Critical 错误接受率",
    "near_lure": "Near lure 错误接受率",
    "unrelated_lure": "Unrelated lure 错误接受率",
}
METRIC_IS_HIT = {"target": True, "critical_recombination": False, "near_lure": False, "unrelated_lure": False}

SEED_RE = re.compile(r"^s(\d+)_")


def trial_seed(trial_id: str) -> int:
    m = SEED_RE.match(trial_id)
    if not m:
        raise ValueError(f"bad trial_id: {trial_id}")
    return int(m.group(1))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trial_metric(row: dict) -> float | None:
    """1 = desired outcome for protocol metric; None if unparseable."""
    pred, gold = row["prediction"], row["gold"]
    if pred is None:
        return None
    correct = pred == gold
    pt = row["probe_type"]
    if METRIC_IS_HIT[pt]:
        return 1.0 if correct else 0.0
    return 1.0 if not correct else 0.0  # false acceptance rate contribution


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% CI across seeds (t-based, n>=2)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    if len(arr) < 2:
        return m, m, m
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    half = 1.96 * se  # normal approx for n=18
    return m, m - half, m + half


def factor_levels(tree: dict) -> tuple[list[int], list[int], list[int]]:
    reps = sorted({k[0] for k in tree})
    ovs = sorted({k[1] for k in tree})
    nls = sorted({k[2] for k in tree})
    return reps, ovs, nls


def seed_rates(rows: list[dict]) -> dict[tuple, dict[int, dict[str, list[float]]]]:
    """(rep, ov, nl) -> seed -> probe_type -> list of per-trial metric values."""
    tree: dict[tuple, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        key = (row["repetition"], row["overlap"], row["context_newlines"])
        seed = trial_seed(row["trial_id"])
        v = trial_metric(row)
        if v is not None:
            tree[key][seed][row["probe_type"]].append(v)
    return tree


def seed_level_means(tree, key: tuple, probe_type: str) -> list[float]:
    out = []
    for seed, types in tree[key].items():
        vals = types.get(probe_type, [])
        if vals:
            out.append(float(np.mean(vals)))
    return out


def marginal_seed_means(tree, probe_type: str, axis: int, level: int) -> list[float]:
    """One value per seed: mean of that seed's cell means over the other factors."""
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key, seeds in tree.items():
        if key[axis] != level:
            continue
        for seed, types in seeds.items():
            vals = types.get(probe_type, [])
            if vals:
                by_seed[seed].append(float(np.mean(vals)))
    return [float(np.mean(vs)) for vs in by_seed.values() if vs]


def aggregate_by_probe(rows: list[dict]) -> dict[str, tuple[float, int]]:
    sums: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        v = trial_metric(row)
        if v is not None:
            sums[row["probe_type"]].append(v)
    return {
        pt: (float(np.mean(sums[pt])) if sums[pt] else float("nan"), len(sums[pt]))
        for pt in PROBE_TYPES
    }


def plot_overview(by_probe: dict[str, tuple[float, int]], out: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(PROBE_TYPES))
    rates = [by_probe[pt][0] for pt in PROBE_TYPES]
    colors = ["#2a6f4e", "#c44e52", "#dd8452", "#8172b3"]
    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    _set_cjk_xticklabels(
        ax, [PROBE_LABELS[pt] for pt in PROBE_TYPES], rotation=15, ha="right"
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("比率", **_fp())
    ax.set_title(f"{title_prefix} · Test 集总体（按题型，非总准确率）", **_fp())
    ax.axhline(0.375, color="#888", ls="--", lw=0.8, label="YES 基线占比 3/8（勿作主指标）")
    for bar, (r, n) in zip(bars, [by_probe[pt] for pt in PROBE_TYPES]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{r:.1%}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, prop=CJK_FONT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _series_mean_ci(tree, probe_type: str, axis: int, levels: list[int]) -> tuple[list[float], list[float], list[float]]:
    means, los, his = [], [], []
    for level in levels:
        m, lo, hi = mean_ci(marginal_seed_means(tree, probe_type, axis, level))
        means.append(m)
        los.append(lo)
        his.append(hi)
    return means, los, his


def plot_factor_context(tree, out: Path) -> None:
    """Fix repetition and overlap; vary context_newlines."""
    reps, ovs, nls = factor_levels(tree)
    nrows, ncols = len(reps), len(ovs)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), sharex=True, squeeze=False
    )
    xlabels = [str(nl) for nl in nls]
    for ax, (rep, ov) in zip(axes.flat, [(r, o) for r in reps for o in ovs]):
        t_means, t_lo, t_hi = [], [], []
        c_means, c_lo, c_hi = [], [], []
        for nl in nls:
            key = (rep, ov, nl)
            tm = seed_level_means(tree, key, "target")
            cm = seed_level_means(tree, key, "critical_recombination")
            m, lo, hi = mean_ci(tm)
            t_means.append(m)
            t_lo.append(lo)
            t_hi.append(hi)
            m, lo, hi = mean_ci(cm)
            c_means.append(m)
            c_lo.append(lo)
            c_hi.append(hi)
        xs = np.arange(len(nls))
        ax.errorbar(
            xs,
            t_means,
            yerr=[np.array(t_means) - np.array(t_lo), np.array(t_hi) - np.array(t_means)],
            fmt="o-",
            color="#2a6f4e",
            capsize=4,
            label="目标命中率",
        )
        ax.errorbar(
            xs,
            c_means,
            yerr=[np.array(c_means) - np.array(c_lo), np.array(c_hi) - np.array(c_means)],
            fmt="s--",
            color="#c44e52",
            capsize=4,
            label="关键重组错误接受率",
        )
        ax.set_title(f"rep={rep}, overlap={ov}", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=7, rotation=25, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("context_newlines（数字个数）", **_fp())
    for ax in axes[:, 0]:
        ax.set_ylabel("比率", **_fp())
    fig.suptitle(
        "因素：上下文距离（固定 repetition & overlap；误差=跨 seed 95% CI）",
        y=1.02,
        **_fp(),
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=9,
        prop=CJK_FONT,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_factor_overlap(tree, out: Path) -> None:
    reps, ovs, nls = factor_levels(tree)
    nrows, ncols = len(reps), len(nls)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), sharey=True, squeeze=False
    )
    width = 0.8 / max(len(ovs), 1)
    for ax, (rep, nl) in zip(axes.flat, [(r, n) for r in reps for n in nls]):
        metrics_t, metrics_c = [], []
        err_t, err_c = [], []
        for ov in ovs:
            key = (rep, ov, nl)
            tm = seed_level_means(tree, key, "target")
            cm = seed_level_means(tree, key, "critical_recombination")
            m, lo, hi = mean_ci(tm)
            metrics_t.append(m)
            err_t.append((hi - m, m - lo))
            m, lo, hi = mean_ci(cm)
            metrics_c.append(m)
            err_c.append((hi - m, m - lo))
        xs = np.arange(len(ovs))
        ax.bar(
            xs - width / 2,
            metrics_t,
            width,
            yerr=np.array(err_t).T if err_t else None,
            color="#2a6f4e",
            label="目标命中率",
            capsize=3,
        )
        ax.bar(
            xs + width / 2,
            metrics_c,
            width,
            yerr=np.array(err_c).T if err_c else None,
            color="#c44e52",
            label="关键重组错误接受率",
            capsize=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"ov={ov}" for ov in ovs], fontsize=7)
        ax.set_title(f"rep={rep}, nl={nl}", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("比率", **_fp())
    fig.suptitle("因素：特征重叠（固定 repetition & context_newlines）", y=1.02, **_fp())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        prop=CJK_FONT,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_factor_repetition(tree, out: Path) -> None:
    reps, ovs, nls = factor_levels(tree)
    nrows, ncols = len(ovs), len(nls)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.4 * ncols, 2.6 * nrows), sharey=True, squeeze=False
    )
    width = 0.8 / max(len(reps), 1)
    for ax, (ov, nl) in zip(axes.flat, [(o, n) for o in ovs for n in nls]):
        metrics_t, metrics_c = [], []
        err_t, err_c = [], []
        for rep in reps:
            key = (rep, ov, nl)
            tm = seed_level_means(tree, key, "target")
            cm = seed_level_means(tree, key, "critical_recombination")
            m, lo, hi = mean_ci(tm)
            metrics_t.append(m)
            err_t.append((hi - m, m - lo))
            m, lo, hi = mean_ci(cm)
            metrics_c.append(m)
            err_c.append((hi - m, m - lo))
        xs = np.arange(len(reps))
        ax.bar(
            xs - width / 2,
            metrics_t,
            width,
            yerr=np.array(err_t).T if err_t else None,
            color="#2a6f4e",
            label="目标命中率",
            capsize=3,
        )
        ax.bar(
            xs + width / 2,
            metrics_c,
            width,
            yerr=np.array(err_c).T if err_c else None,
            color="#c44e52",
            label="关键重组错误接受率",
            capsize=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"rep={rep}" for rep in reps], fontsize=7)
        ax.set_title(f"overlap={ov}, nl={nl}", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("比率", **_fp())
    fig.suptitle("因素：特征重复（固定 overlap & context_newlines）", y=1.02, **_fp())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        prop=CJK_FONT,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_factor_context_avg(tree, out: Path, title_prefix: str) -> None:
    """Average repetition and overlap; vary context_newlines."""
    _, _, nls = factor_levels(tree)
    t_means, t_lo, t_hi = _series_mean_ci(tree, "target", 2, nls)
    c_means, c_lo, c_hi = _series_mean_ci(tree, "critical_recombination", 2, nls)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(nls))
    ax.errorbar(
        xs,
        t_means,
        yerr=[np.array(t_means) - np.array(t_lo), np.array(t_hi) - np.array(t_means)],
        fmt="o-",
        color="#2a6f4e",
        capsize=4,
        label="目标命中率",
    )
    ax.errorbar(
        xs,
        c_means,
        yerr=[np.array(c_means) - np.array(c_lo), np.array(c_hi) - np.array(c_means)],
        fmt="s--",
        color="#c44e52",
        capsize=4,
        label="关键重组错误接受率",
    )
    ax.set_xticks(xs)
    _set_cjk_xticklabels(ax, [str(nl) for nl in nls])
    ax.set_xlabel("上下文距离（数字个数）", **_fp())
    ax.set_ylabel("比率", **_fp())
    ax.set_ylim(-0.05, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{title_prefix} · 上下文距离（其余因素已平均；误差=跨 seed 95% CI）", **_fp())
    ax.legend(loc="best", prop=CJK_FONT)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_marginal_bars(
    tree, out: Path, axis: int, levels: list[int], xlabels: list[str], xlabel: str, title: str
) -> None:
    t_means, t_lo, t_hi = _series_mean_ci(tree, "target", axis, levels)
    c_means, c_lo, c_hi = _series_mean_ci(tree, "critical_recombination", axis, levels)
    err_t = (np.array(t_hi) - np.array(t_means), np.array(t_means) - np.array(t_lo))
    err_c = (np.array(c_hi) - np.array(c_means), np.array(c_means) - np.array(c_lo))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.arange(len(levels))
    width = 0.36
    ax.bar(xs - width / 2, t_means, width, yerr=err_t, color="#2a6f4e", label="目标命中率", capsize=4)
    ax.bar(xs + width / 2, c_means, width, yerr=err_c, color="#c44e52", label="关键重组错误接受率", capsize=4)
    ax.set_xticks(xs)
    _set_cjk_xticklabels(ax, xlabels)
    ax.set_xlabel(xlabel, **_fp())
    ax.set_ylabel("比率", **_fp())
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, **_fp())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, prop=CJK_FONT)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_factor_overlap_avg(tree, out: Path, title_prefix: str) -> None:
    _, ovs, _ = factor_levels(tree)
    _plot_marginal_bars(
        tree,
        out,
        axis=1,
        levels=ovs,
        xlabels=[f"overlap={ov}" for ov in ovs],
        xlabel="特征重叠",
        title=f"{title_prefix} · 特征重叠（其余因素已平均；误差=跨 seed 95% CI）",
    )


def plot_factor_repetition_avg(tree, out: Path, title_prefix: str) -> None:
    reps, _, _ = factor_levels(tree)
    _plot_marginal_bars(
        tree,
        out,
        axis=0,
        levels=reps,
        xlabels=[f"rep={rep}" for rep in reps],
        xlabel="特征重复",
        title=f"{title_prefix} · 特征重复（其余因素已平均；误差=跨 seed 95% CI）",
    )


def plot_marginal_factors(tree, output_dir: Path, title_prefix: str) -> None:
    plot_factor_context_avg(tree, output_dir / "07_factor_context_newlines_avg.png", title_prefix)
    plot_factor_overlap_avg(tree, output_dir / "08_factor_overlap_avg.png", title_prefix)
    plot_factor_repetition_avg(tree, output_dir / "09_factor_repetition_avg.png", title_prefix)


# Short legend names. Colors stay with the model across every panel.
CROSS_MODEL_SPECS = (
    ("qwen3vl-4b", "Qwen3-VL-4B", "#4C78A8", "o"),
    ("qwen3vl-8b", "Qwen3-VL-8B", "#F58518", "s"),
    ("kimi-vl-a3b", "Kimi-VL-A3B", "#54A24B", "D"),
    ("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash", "#B279A2", "^"),
)
_CROSS_PANELS = (
    (2, "Context distance (digit count)", "Context distance", None),
    (1, "Overlap", "Overlap", None),
    (0, "Repetition", "Repetition", None),
)


def _ylim_from_errorbars(los: list[float], his: list[float]) -> tuple[float, float]:
    """Pad the observed CI range. Do not pin the axis to 0–1."""
    lo = float(np.min(los))
    hi = float(np.max(his))
    span = hi - lo
    if span < 1e-6:
        span = 0.02
    pad = max(span * 0.15, 0.008)
    ymin, ymax = lo - pad, hi + pad
    if lo >= 0 and ymin < 0:
        ymin = 0.0
    return ymin, ymax


def plot_cross_model_factors(results_root: Path, output_dir: Path) -> None:
    """Six figures: three factors × two protocol metrics, four models overlaid.

    Non-contrast factors are averaged within each seed, then the mean and 95% CI
    are taken across seeds. Each panel scales its own y-axis to the data. Models
    are not pooled.
    """
    trees = []
    for dirname, _label, _color, _marker in CROSS_MODEL_SPECS:
        rows = [r for r in load_rows(results_root / dirname / "predictions.jsonl") if r.get("split") == "test"]
        trees.append(seed_rates(rows))
    level_sets = [factor_levels(tree) for tree in trees]
    if any(levels != level_sets[0] for levels in level_sets[1:]):
        raise ValueError("factor levels differ across models; refusing to overlay")
    reps, ovs, nls = level_sets[0]
    levels_by_axis = {0: reps, 1: ovs, 2: nls}
    metrics = (
        ("target", "Target hit rate", "target"),
        ("critical_recombination", "Critical false-acceptance rate", "critical"),
    )
    filenames = {
        (2, "target"): "01_context_target.png",
        (2, "critical"): "02_context_critical.png",
        (1, "target"): "03_overlap_target.png",
        (1, "critical"): "04_overlap_critical.png",
        (0, "target"): "05_repetition_target.png",
        (0, "critical"): "06_repetition_critical.png",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(CROSS_MODEL_SPECS)
    bar_width = 0.18
    for axis, xlabel, short_name, _label_fmt in _CROSS_PANELS:
        levels = levels_by_axis[axis]
        tick_labels = [str(level) for level in levels]
        xs = np.arange(len(levels))
        crowded = len(levels) > 3
        for probe, metric_name, metric_key in metrics:
            fig, ax = plt.subplots(figsize=(18.4 if crowded else 11.2, 6.4))
            all_lo, all_hi = [], []
            series = []
            for i, ((_dirname, label, color, _marker), tree) in enumerate(zip(CROSS_MODEL_SPECS, trees)):
                means, los, his = _series_mean_ci(tree, probe, axis, levels)
                all_lo.extend(los)
                all_hi.extend(his)
                series.append((i, label, color, means, los, his))
            ymin, ymax = _ylim_from_errorbars(all_lo, all_hi)
            span = ymax - ymin
            headroom = span * 0.16
            ax.set_ylim(ymin, ymax + headroom)
            ax.set_xlim(-0.55, len(levels) - 0.45)
            for i, label, color, means, los, his in series:
                shift = (i - (n_models - 1) / 2) * bar_width
                yerr = [np.asarray(means) - np.asarray(los), np.asarray(his) - np.asarray(means)]
                ax.bar(
                    xs + shift,
                    means,
                    bar_width * 0.92,
                    yerr=yerr,
                    color=color,
                    label=label,
                    capsize=2,
                    error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
                    zorder=2,
                )
                for x, mean, hi in zip(xs, means, his):
                    ax.text(
                        x + shift,
                        hi + span * 0.012,
                        f"{mean * 100:.2f}%",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=7 if crowded else 8,
                        color="#222222",
                        clip_on=False,
                        zorder=3,
                    )
            ax.set_xticks(xs)
            ax.set_xticklabels(tick_labels, fontsize=15)
            ax.tick_params(axis="y", labelsize=15)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(metric_name, fontsize=16)
            ax.set_title(
                f"{short_name}: {metric_name.lower()}\n"
                "Other factors averaged within each seed. Error bars: 95% CI across seeds.",
                fontsize=18,
            )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=15)
            fig.tight_layout()
            fig.savefig(output_dir / filenames[(axis, metric_key)], dpi=160, bbox_inches="tight")
            plt.close(fig)


def plot_heatmap(tree, out: Path) -> None:
    keys = sorted(tree.keys())
    labels = [f"r{k[0]}_o{k[1]}_nl{k[2]}" for k in keys]
    data = np.zeros((len(keys), len(PROBE_TYPES)))
    for i, key in enumerate(keys):
        for j, pt in enumerate(PROBE_TYPES):
            data[i, j] = float(np.mean(seed_level_means(tree, key, pt))) if seed_level_means(tree, key, pt) else np.nan
    fig_h = max(7, 0.22 * len(keys))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(PROBE_TYPES)))
    _set_cjk_xticklabels(
        ax,
        [PROBE_LABELS[pt].replace(" ", "\n") for pt in PROBE_TYPES],
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(keys)):
        for j in range(len(PROBE_TYPES)):
            ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center", color="black", fontsize=7)
    ax.set_title(f"{len(keys)} 组实验条件 × 四类题型指标（跨 seed 均值）", **_fp())
    cbar = fig.colorbar(im, ax=ax, label="比率")
    if CJK_FONT is not None:
        cbar.set_label("比率", fontproperties=CJK_FONT)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_binding_dissociation(tree, out: Path) -> None:
    """Target vs critical per condition — binding error signature."""
    keys = sorted(tree.keys())
    fig, ax = plt.subplots(figsize=(7, 6))
    for key in keys:
        t = float(np.mean(seed_level_means(tree, key, "target")))
        c = float(np.mean(seed_level_means(tree, key, "critical_recombination")))
        ax.scatter(t, c, s=50, alpha=0.85)
        ax.annotate(f"r{key[0]}o{key[1]}n{key[2]}", (t, c), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Target 命中率（跨 seed 均值）", **_fp())
    ax.set_ylabel("Critical 错误接受率", **_fp())
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(1, color="#ccc", lw=0.5, ls=":")
    ax.set_title("遗忘 vs 绑定：理想绑定错误在 target 稳定时 critical 升高", **_fp())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_report_md(by_probe, tree, out: Path, n_test: int, title: str) -> None:
    lines = [
        f"# {title} · OBM-VLM-Task2 结果汇总",
        "",
        "## 图表设计（对应 README 协议）",
        "",
        "1. **题型总览条图**：只报告 target / critical / near / unrelated 四类指标，不用总准确率。",
        "2. **三因素分面图**（02–04）：每次固定另外两个因素，单独考察 context_newlines、overlap、repetition。",
        "3. **三因素平均图**（07–09）：每张图只变化一个因素，另外两个因素在每个 seed 内先取平均。",
        "4. **全条件热图**：展示全因子网格，便于对照交互。",
        "5. **Target–Critical 散点**：区分「一般遗忘」（target 降）与「绑定错误」（target 稳、critical 升）。",
        "",
        "误差条：分面图在每个完整条件下先算每个 test seed 的题型均值，再对 18 个 seed 求均值与近似 95% CI。平均图先在每个 seed 内对非对比因素取平均，再跨 seed 计算。热图与散点仍按完整条件、跨 seed 计算。",
        "",
        f"## 总体（test, n={n_test}）",
        "",
        "| 题型 | 指标 | n |",
        "|------|------|---|",
    ]
    for pt in PROBE_TYPES:
        r, n = by_probe[pt]
        lines.append(f"| {pt} | {r:.4f} | {n} |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).resolve().parent / "results/qwen3vl-8b/predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results/qwen3vl-8b/figures",
    )
    parser.add_argument(
        "--title",
        default="Qwen3-VL-8B-Instruct",
        help="Model name used in the report title and overview figure.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    font = configure_matplotlib_cjk()
    if font:
        print(f"matplotlib CJK font: {font.get_name()}", flush=True)
    else:
        print("warning: no CJK font found; Chinese labels may not render", flush=True)

    rows = [r for r in load_rows(args.predictions) if r.get("split") == "test"]
    tree = seed_rates(rows)
    by_probe = aggregate_by_probe(rows)

    write_report_md(by_probe, tree, args.output_dir.parent / "REPORT.md", len(rows), args.title)
    plot_overview(by_probe, args.output_dir / "01_overview_by_probe_type.png", args.title)
    plot_factor_context(tree, args.output_dir / "02_factor_context_newlines.png")
    plot_factor_overlap(tree, args.output_dir / "03_factor_overlap.png")
    plot_factor_repetition(tree, args.output_dir / "04_factor_repetition.png")
    plot_marginal_factors(tree, args.output_dir, args.title)
    plot_heatmap(tree, args.output_dir / "05_heatmap_12_conditions.png")
    plot_binding_dissociation(tree, args.output_dir / "06_target_vs_critical.png")
    print(f"Wrote figures to {args.output_dir} and REPORT.md")


if __name__ == "__main__":
    main()
