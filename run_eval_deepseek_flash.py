"""Evaluate Task 2 with DeepSeek-V4.1-Flash (thinking disabled).

Set DEEPSEEK_API_KEY in the environment. Do not write the key into this file.

    export DEEPSEEK_API_KEY=...
    python run_eval_deepseek_flash.py --split dev --limit 8
    python run_eval_deepseek_flash.py --split test
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from dataset_api import Task2Dataset, parse_yes_no

# 只从环境变量读取。不要把 key 写进仓库。
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"  # DeepSeek-V4.1-Flash
PROBE_TYPES = ("target", "critical_recombination", "near_lure", "unrelated_lure")
ROOT = Path(__file__).resolve().parent


def content_blocks(dataset: Task2Dataset, trial_id: str, cache: dict[str, str]) -> list[dict]:
    blocks: list[dict] = []
    for item in dataset.qwen_content(trial_id):
        if item["type"] == "text":
            blocks.append({"type": "text", "text": item["text"]})
            continue
        path = item["image"]
        url = cache.get(path)
        if url is None:
            raw = Path(path).read_bytes()
            url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
            cache[path] = url
        blocks.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "original"},
        })
    return blocks


def call_once(client: OpenAI, blocks: list[dict], max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": blocks}],
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        raise RuntimeError("thinking mode still returned reasoning_content")
    return message.content or ""


def call_with_retry(client: OpenAI, blocks: list[dict], max_tokens: int, retries: int) -> str:
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return call_once(client, blocks, max_tokens)
        except Exception as exc:  # noqa: BLE001 — retry transient API errors
            last = exc
            if attempt + 1 == retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    assert last is not None
    raise last


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["trial_id"])
    return done


def summarize(dataset: Task2Dataset, rows: list[dict]) -> dict:
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_condition: dict[tuple, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    parse_fail = 0
    for row in rows:
        label = dataset.get_label(row["trial_id"])
        pred = row["prediction"]
        if pred is None:
            parse_fail += 1
            continue
        correct = pred == label["answer"]
        by_type[label["probe_type"]].append(correct)
        key = (label["repetition"], label["overlap"], label["context_newlines"])
        by_condition[key][label["probe_type"]].append(correct)

    def rate(values: list[bool], positive_is_hit: bool) -> float | None:
        if not values:
            return None
        hit = sum(values) / len(values)
        return hit if positive_is_hit else 1.0 - hit

    summary: dict = {"n_trials": len(rows), "parse_fail": parse_fail, "by_probe_type": {}}
    for pt in PROBE_TYPES:
        hits = by_type.get(pt, [])
        summary["by_probe_type"][pt] = {"n": len(hits), "rate": rate(hits, pt == "target")}
    summary["by_condition"] = {}
    for key, types in sorted(by_condition.items()):
        rep, ov, nl = key
        cond = {"repetition": rep, "overlap": ov, "context_newlines": nl, "probe_types": {}}
        for pt in PROBE_TYPES:
            hits = types.get(pt, [])
            cond["probe_types"][pt] = {"n": len(hits), "rate": rate(hits, pt == "target")}
        summary["by_condition"][f"rep{rep}_ov{ov}_nl{nl}"] = cond
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "deepseek-v4.1-flash",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 道未完成的题")
    parser.add_argument("--workers", type=int, default=8, help="并发请求数")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    if not API_KEY.strip():
        raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY，不要把 key 写进文件")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.output_dir / "predictions.jsonl"
    dataset = Task2Dataset(args.dataset)
    split_filter = None if args.split == "all" else args.split
    done = load_done(preds_path)
    trials = sorted(dataset.iter_examples(split=split_filter), key=lambda row: row["trial_id"])
    pending = [row for row in trials if row["trial_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"model={MODEL} thinking=disabled split={args.split} "
        f"total={len(trials)} done={len(done)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    image_cache: dict[str, str] = {}
    cache_lock = threading.Lock()
    write_lock = threading.Lock()

    def one(example: dict) -> dict:
        trial_id = example["trial_id"]
        with cache_lock:
            blocks = content_blocks(dataset, trial_id, image_cache)
        try:
            raw = call_with_retry(client, blocks, args.max_tokens, args.retries)
            err = None
        except Exception as exc:  # noqa: BLE001 — record and continue
            raw = ""
            err = repr(exc)
        label = dataset.get_label(trial_id)
        return {
            "trial_id": trial_id,
            "split": example["split"],
            "repetition": example["repetition"],
            "overlap": example["overlap"],
            "context_newlines": example["context_newlines"],
            "probe_index": example["probe_index"],
            "probe_type": label["probe_type"],
            "gold": label["answer"],
            "raw": raw,
            "prediction": parse_yes_no(raw),
            "error": err,
        }

    with preds_path.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, example) for example in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="eval"):
                record = future.result()
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with write_lock:
                    out_f.write(line)
                    out_f.flush()

    rows = [
        json.loads(line)
        for line in preds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split_filter is not None:
        rows = [row for row in rows if row["split"] == split_filter]
    summary = summarize(dataset, rows)
    summary["model"] = MODEL
    summary["thinking"] = "disabled"
    summary["split"] = args.split
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_probe_type"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {preds_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
