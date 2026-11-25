"""Task 2 evaluation for LLaVA-NeXT (LLaVA-1.6) Hugging Face checkpoints.

Same protocol as the Qwen runner: test split, greedy decoding, strict YES/NO,
one fresh context per trial.

Long contexts do not fit in one 48GB GPU. Launch two workers: shard 0 sees
CUDA 0,1 and shard 1 sees CUDA 2,3. Each worker slices one model across its
two visible GPUs (device_map=balanced) and scores half of the trials.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

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
            # OOM and empty decodes are not finished trials; a later run retries them.
            if row.get("error") or row.get("prediction") is None:
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


def build_messages(content: list[dict]) -> tuple[list[dict], list[Image.Image]]:
    blocks: list[dict] = []
    images: list[Image.Image] = []
    for item in content:
        if item["type"] == "text":
            blocks.append({"type": "text", "text": item["text"]})
        else:
            image = Image.open(item["image"]).convert("RGB")
            images.append(image)
            blocks.append({"type": "image", "image": image})
    return [{"role": "user", "content": blocks}], images


def _place_inputs(inputs, model):
    # A layer-sliced model has no single device; inputs enter on the first visible GPU.
    if getattr(model, "hf_device_map", None):
        return inputs.to("cuda:0")
    return inputs.to(model.device)


def run_one(model, processor, content: list[dict]) -> str:
    messages, images = build_messages(content)
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _place_inputs(inputs, model)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        texts = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return texts[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "Dataset")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help=(
            "single: one full copy on cuda:0. "
            "auto: slice one copy across every visible GPU (use with CUDA_VISIBLE_DEVICES=0,1 or 2,3)."
        ),
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

    preds_path = (
        args.output_dir / f"predictions.shard{args.shard_id}.jsonl"
        if args.num_shards > 1
        else args.output_dir / "predictions.jsonl"
    )
    if args.num_shards > 1 and args.shard_id is None:
        raise SystemExit("set --shard-id when --num-shards > 1")

    done = load_done_trial_ids(args.output_dir, args.num_shards)
    n_gpu = torch.cuda.device_count()
    print(
        f"Loading LLaVA-NeXT from {args.model} device_map={args.device_map} visible_gpus={n_gpu}",
        flush=True,
    )
    load_kwargs: dict = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if args.device_map == "auto":
        if n_gpu < 2:
            raise SystemExit("--device-map auto needs at least two visible GPUs")
        # "balanced" splits layers across both cards. "auto" would fill GPU 0 first
        # and leave the second card idle, which is how the single-card OOM happened.
        load_kwargs["device_map"] = "balanced"
        load_kwargs["max_memory"] = {i: "40GiB" for i in range(n_gpu)}
    model = LlavaNextForConditionalGeneration.from_pretrained(str(args.model), **load_kwargs)
    if args.device_map == "single":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    processor = LlavaNextProcessor.from_pretrained(str(args.model))
    model.eval()
    if getattr(model, "hf_device_map", None):
        print(f"hf_device_map devices={sorted(set(map(str, model.hf_device_map.values())))}", flush=True)

    trials = sorted(dataset.iter_examples(split=split_filter), key=lambda row: row["trial_id"])
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

    with preds_path.open("a", encoding="utf-8") as out_f:
        for example in tqdm(pending, desc="eval"):
            trial_id = example["trial_id"]
            content = dataset.qwen_content(trial_id)
            try:
                raw = run_one(model, processor, content)
                err = None
            except Exception as exc:  # noqa: BLE001
                raw = ""
                err = repr(exc)
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

    if args.num_shards > 1:
        print(f"Shard {args.shard_id} finished; wrote {preds_path}", flush=True)
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


if __name__ == "__main__":
    main()
