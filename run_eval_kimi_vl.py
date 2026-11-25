"""Task 2 evaluation for Kimi-VL-A3B-Instruct.

Same protocol as the Qwen runner: test split, greedy decoding, strict YES/NO,
one fresh context per trial. truncation is off so the 128K window is actually used.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from dataset_api import Task2Dataset, parse_yes_no

PROBE_TYPES = ("target", "critical_recombination", "near_lure", "unrelated_lure")


def load_done_trial_ids(output_dir: Path, num_shards: int) -> set[str]:
    done: set[str] = set()
    paths = [output_dir / "predictions.jsonl"]
    if num_shards > 1:
        paths = [output_dir / f"predictions.shard{i}.jsonl" for i in range(num_shards)]
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            done.add(row["trial_id"])
    return done


def merge_shard_predictions(output_dir: Path, num_shards: int) -> Path:
    rows: dict[str, dict] = {}
    for i in range(num_shards):
        path = output_dir / f"predictions.shard{i}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["trial_id"]] = row
    merged = output_dir / "predictions.jsonl"
    with merged.open("w", encoding="utf-8") as out_f:
        for trial_id in sorted(rows):
            out_f.write(json.dumps(rows[trial_id], ensure_ascii=False) + "\n")
    return merged


def _place_inputs(inputs, model):
    device = "cuda:0" if getattr(model, "hf_device_map", None) else model.device
    if hasattr(inputs, "to"):
        return inputs.to(device)
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def run_one(model, processor, content: list[dict]) -> str:
    messages = [{"role": "user", "content": content}]
    images: list[Image.Image] = []
    try:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        for item in content:
            if item["type"] == "image":
                images.append(Image.open(item["image"]).convert("RGB"))
        inputs = processor(
            images=images,
            text=text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = _place_inputs(inputs, model)
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        trimmed = sequences[:, input_len:]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    finally:
        for image in images:
            image.close()


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


def load_model(model_path: Path, device_map: str):
    # Remote config accepts only eager and flash_attention_2. sdpa raises
    # ValueError, and flash_attn is not installed in this environment.
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if device_map == "auto":
        n_gpu = torch.cuda.device_count()
        if n_gpu < 2:
            raise SystemExit("--device-map auto needs at least two visible GPUs")
        kwargs["device_map"] = "balanced"
        kwargs["max_memory"] = {i: "40GiB" for i in range(n_gpu)}
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    if device_map == "single":
        model.to(torch.device("cuda:0"))
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "Dataset")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-newlines", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help="single: one copy on cuda:0. auto: slice one copy across the visible GPUs.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Task2Dataset(args.dataset)
    split_filter = None if args.split == "all" else args.split
    summary_path = args.output_dir / "summary.json"

    if args.merge_only:
        preds_path = merge_shard_predictions(args.output_dir, args.num_shards)
        all_rows = [
            json.loads(line)
            for line in preds_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if split_filter is not None:
            all_rows = [row for row in all_rows if row["split"] == split_filter]
        summary = summarize(dataset, all_rows)
        summary["model"] = str(args.model)
        summary["split"] = args.split
        summary["num_shards"] = args.num_shards
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary["by_probe_type"], ensure_ascii=False, indent=2), flush=True)
        print(f"Merged {preds_path} and wrote {summary_path}", flush=True)
        return

    if args.num_shards > 1 and args.shard_id is None:
        raise SystemExit("set --shard-id when --num-shards > 1")
    preds_path = (
        args.output_dir / f"predictions.shard{args.shard_id}.jsonl"
        if args.num_shards > 1
        else args.output_dir / "predictions.jsonl"
    )

    done = load_done_trial_ids(args.output_dir, args.num_shards)
    print(
        f"Loading Kimi-VL from {args.model} device_map={args.device_map} "
        f"visible_gpus={torch.cuda.device_count()}",
        flush=True,
    )
    model, processor = load_model(args.model, args.device_map)
    if getattr(model, "hf_device_map", None):
        print(
            f"hf_device_map devices={sorted(set(map(str, model.hf_device_map.values())))}",
            flush=True,
        )

    trials = sorted(dataset.iter_examples(split=split_filter, context_newlines=args.context_newlines), key=lambda row: row["trial_id"])
    if args.num_shards > 1:
        trials = [row for i, row in enumerate(trials) if i % args.num_shards == args.shard_id]
    if args.limit is not None:
        trials = trials[: args.limit]
    pending = [row for row in trials if row["trial_id"] not in done]
    print(
        f"split={args.split} shard={args.shard_id}/{args.num_shards} "
        f"total={len(trials)} done={len(done)} pending={len(pending)}",
        flush=True,
    )

    failures = 0
    with preds_path.open("a", encoding="utf-8") as out_f:
        for example in tqdm(pending, desc="eval"):
            trial_id = example["trial_id"]
            content = dataset.qwen_content(trial_id)
            try:
                raw = run_one(model, processor, content)
                err = None
            except Exception as exc:  # noqa: BLE001
                raw = ""
                err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                failures += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            label = dataset.get_label(trial_id)
            record = {
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
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            if err and args.fail_on_error:
                if "out of memory" in err.lower():
                    raise SystemExit(2)
                raise SystemExit(err)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"peak_reserved_gib={peak:.2f}", flush=True)

    if args.num_shards > 1:
        print(f"Shard {args.shard_id} finished; wrote {preds_path}", flush=True)
        if args.fail_on_error and failures:
            raise SystemExit(f"{failures} trial(s) failed")
        return

    all_rows = [
        json.loads(line)
        for line in preds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split_filter is not None:
        all_rows = [row for row in all_rows if row["split"] == split_filter]
    summary = summarize(dataset, all_rows)
    summary["model"] = str(args.model)
    summary["split"] = args.split
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_probe_type"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {preds_path} and {summary_path}", flush=True)
    if args.fail_on_error and failures:
        raise SystemExit(f"{failures} trial(s) failed")


if __name__ == "__main__":
    main()
