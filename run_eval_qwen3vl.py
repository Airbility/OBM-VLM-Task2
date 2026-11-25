"""Full Task 2 evaluation with Qwen3-VL (protocol-aligned)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
            if line.strip():
                done.add(json.loads(line)["trial_id"])
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
    # Sharded models (device_map="auto") take inputs on the first visible GPU.
    if getattr(model, "hf_device_map", None):
        return inputs.to("cuda:0")
    return inputs.to(model.device)


def run_one(model, processor, content: list[dict]) -> str:
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
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


def summarize(dataset: Task2Dataset, rows: list[dict]) -> dict:
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_condition: dict[tuple, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    parse_fail = 0
    for row in rows:
        label = dataset.get_label(row["trial_id"])
        pred = row["prediction"]
        probe_type = label["probe_type"]
        gold = label["answer"]
        if pred is None:
            parse_fail += 1
            continue
        correct = pred == gold
        by_type[probe_type].append(correct)
        key = (
            label["repetition"],
            label["overlap"],
            label["context_newlines"],
        )
        by_condition[key][probe_type].append(correct)

    def rate(correct_list: list[bool], positive_is_hit: bool) -> float | None:
        if not correct_list:
            return None
        if positive_is_hit:
            return sum(correct_list) / len(correct_list)
        return 1.0 - (sum(correct_list) / len(correct_list))

    summary: dict = {
        "n_trials": len(rows),
        "parse_fail": parse_fail,
        "by_probe_type": {},
    }
    for pt in PROBE_TYPES:
        hits = by_type.get(pt, [])
        positive_is_hit = pt == "target"
        summary["by_probe_type"][pt] = {
            "n": len(hits),
            "rate": rate(hits, positive_is_hit),
        }

    summary["by_condition"] = {}
    for key, types in sorted(by_condition.items()):
        rep, ov, nl = key
        cond = {
            "repetition": rep,
            "overlap": ov,
            "context_newlines": nl,
            "probe_types": {},
        }
        for pt in PROBE_TYPES:
            hits = types.get(pt, [])
            positive_is_hit = pt == "target"
            cond["probe_types"][pt] = {
                "n": len(hits),
                "rate": rate(hits, positive_is_hit),
            }
        summary["by_condition"][f"rep{rep}_ov{ov}_nl{nl}"] = cond
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "Dataset",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("dev", "test", "all"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "qwen3vl-8b",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: max trials")
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard index 0..num-shards-1 for parallel single-GPU workers.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of parallel shards (each should use one GPU).",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge predictions.shard*.jsonl into predictions.jsonl and write summary.",
    )
    parser.add_argument(
        "--device-map",
        choices=("single", "auto"),
        default="single",
        help="single: one full model on cuda:0. auto: shard layers across visible GPUs.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    dataset = Task2Dataset(args.dataset)
    split_filter = None if args.split == "all" else args.split

    if args.merge_only:
        preds_path = merge_shard_predictions(args.output_dir, args.num_shards)
        all_rows = [
            json.loads(line)
            for line in preds_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if split_filter is not None:
            all_rows = [r for r in all_rows if r["split"] == split_filter]
        summary = summarize(dataset, all_rows)
        summary["model"] = str(args.model)
        summary["split"] = args.split
        summary["num_shards"] = args.num_shards
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary["by_probe_type"], ensure_ascii=False, indent=2), flush=True)
        print(f"Merged {preds_path} and wrote {summary_path}", flush=True)
        return

    if args.num_shards > 1:
        if args.shard_id is None:
            raise SystemExit("set --shard-id when --num-shards > 1")
        preds_path = args.output_dir / f"predictions.shard{args.shard_id}.jsonl"
    else:
        preds_path = args.output_dir / "predictions.jsonl"

    done = load_done_trial_ids(args.output_dir, args.num_shards)

    if args.device_map == "auto" and args.num_shards > 1:
        raise SystemExit("--device-map auto uses every visible GPU; do not combine with shards")

    print(f"Loading model from {args.model} device_map={args.device_map}", flush=True)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if args.device_map == "auto":
        n_gpu = torch.cuda.device_count()
        # Leave room for the memory already in use on these cards.
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {i: "40GiB" for i in range(n_gpu)}
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model), **load_kwargs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device_map == "single":
        model.to(device)
    processor = AutoProcessor.from_pretrained(str(args.model))
    model.eval()

    trials = sorted(
        list(dataset.iter_examples(split=split_filter)),
        key=lambda t: t["trial_id"],
    )
    if args.num_shards > 1:
        trials = [
            t for i, t in enumerate(trials) if i % args.num_shards == args.shard_id
        ]
    if args.limit is not None:
        trials = trials[: args.limit]

    pending = [t for t in trials if t["trial_id"] not in done]
    shard_tag = f" shard={args.shard_id}/{args.num_shards}" if args.num_shards > 1 else ""
    print(
        f"split={args.split}{shard_tag} device={device} "
        f"total={len(trials)} done={len(done)} pending={len(pending)}",
        flush=True,
    )

    with preds_path.open("a", encoding="utf-8") as out_f:
        for example in tqdm(pending, desc="eval"):
            trial_id = example["trial_id"]
            content = dataset.qwen_content(trial_id)
            try:
                raw = run_one(model, processor, content)
            except Exception as exc:  # noqa: BLE001 — log and continue
                raw = ""
                err = repr(exc)
            else:
                err = None
            prediction = parse_yes_no(raw)
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
                "prediction": prediction,
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
        all_rows = [r for r in all_rows if r["split"] == split_filter]

    summary = summarize(dataset, all_rows)
    summary["model"] = str(args.model)
    summary["split"] = args.split
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["by_probe_type"], ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {preds_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
