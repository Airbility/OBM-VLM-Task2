"""Write images, prompts, and private labels for the Task 2 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from obm_task2.constants import (
    CONTEXT_NEWLINES,
    DATASET_VERSION,
    DISTRACTOR_COUNT,
    DISTRACTOR_CUE,
    FILLER_PREFIX,
    FILLER_SUFFIX,
    INSTRUCTION,
    OVERLAPS,
    REPETITIONS,
    STUDY_COUNT,
    TEST_CUE,
    filler_text,
)
from obm_task2.design import iter_trials
from obm_task2.render import opaque_name, probe_variant, save_object
from obm_task2.validate import validate_dataset


ROOT = Path(__file__).resolve().parent


def _image_file(kind: str, obj, probe_id: str | None = None) -> str:
    if kind == "probe":
        key = f"probe|{probe_id}|{obj.key()}"
    else:
        key = f"{kind}|{obj.key()}"
    return f"images/{opaque_name(key)}.png"


def _messages(
    study_files: list[str],
    distractor_files: list[str],
    probe_file: str,
    trial,
) -> list[dict]:
    messages = [{"type": "text", "text": INSTRUCTION}]
    messages.extend({"type": "image", "path": path} for path in study_files)
    messages.append({
        "type": "text",
        "text": filler_text(
            trial.context_newlines,
            seed=trial.seed,
            repetition=trial.repetition,
            overlap=trial.overlap,
        ),
    })
    messages.append({"type": "text", "text": DISTRACTOR_CUE})
    messages.extend({"type": "image", "path": path} for path in distractor_files)
    messages.append({"type": "text", "text": TEST_CUE})
    messages.append({"type": "image", "path": probe_file})
    return messages


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(output: Path) -> None:
    image_dir = output / "images"
    prompt_dir = output / "prompts"
    label_dir = output / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    trials = iter_trials()
    rendered: set[str] = set()
    manifest = []
    labels = []

    def render(kind: str, obj, probe_id: str | None = None) -> str:
        relative = _image_file(kind, obj, probe_id)
        if relative not in rendered:
            variant = probe_variant(probe_id) if kind == "probe" else None
            save_object(image_dir / Path(relative).name, obj.color, obj.shape, obj.texture, variant)
            rendered.add(relative)
        return relative

    for index, trial in enumerate(trials, start=1):
        study_files = [render("study", obj) for obj in trial.study_objects]
        distractor_files = [render("distractor", obj) for obj in trial.distractor_objects]
        probe_file = render("probe", trial.probe_object, trial.probe_id)
        messages = _messages(study_files, distractor_files, probe_file, trial)
        image_paths = [item["path"] for item in messages if item["type"] == "image"]
        prompt = {
            "trial_id": trial.trial_id,
            "image_paths": image_paths,
            "messages": messages,
        }
        _write_json(prompt_dir / f"{trial.trial_id}.json", prompt)
        label = trial.label_row()
        label.update({
            "study_images": study_files,
            "distractor_images": distractor_files,
            "test_image": probe_file,
            "prompt_file": f"prompts/{trial.trial_id}.json",
        })
        labels.append(label)
        manifest.append({
            "trial_id": trial.trial_id,
            "split": trial.split,
            "seed": trial.seed,
            "repetition": trial.repetition,
            "overlap": trial.overlap,
            "context_newlines": trial.context_newlines,
            "probe_index": trial.probe_index,
            "prompt_file": f"prompts/{trial.trial_id}.json",
            "image_paths": image_paths,
        })
        if index % 200 == 0:
            print(f"wrote {index}/{len(trials)} prompts", flush=True)

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (label_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    metadata = {
        "dataset_version": DATASET_VERSION,
        "image_size": [384, 384],
        "study_count": STUDY_COUNT,
        "distractor_count": DISTRACTOR_COUNT,
        "repetitions": list(REPETITIONS),
        "overlaps": list(OVERLAPS),
        "context_newlines": list(CONTEXT_NEWLINES),
        "seeds": 24,
        "dev_seeds": "1-6",
        "test_seeds": "7-24",
        "probes_per_condition": {
            "target": 3,
            "critical_recombination": 3,
            "near_lure": 1,
            "unrelated_lure": 1,
        },
        "trial_count": len(trials),
        "image_count": len(rendered),
        "filler_prefix": FILLER_PREFIX,
        "filler_suffix": FILLER_SUFFIX,
        "filler_format": "prefix + newline + n_random_digits + newline + suffix",
        "filler_digit_levels": list(CONTEXT_NEWLINES),
        "python": sys.version,
    }
    _write_json(output / "metadata.json", metadata)
    errors = validate_dataset(output)
    if errors:
        raise SystemExit("validation failed:\n" + "\n".join(errors[:30]))
    print(f"images: {len(rendered)}; prompts: {len(trials)}; validation: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Dataset")
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
