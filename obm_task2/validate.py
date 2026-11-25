"""Checks that prompts, labels, and images implement the three factors."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from obm_task2.constants import (
    CONTEXT_NEWLINES,
    DISTRACTOR_COUNT,
    DISTRACTOR_CUE,
    EXTRA_COLORS,
    EXTRA_SHAPES,
    EXTRA_TEXTURES,
    FILLER_PREFIX,
    FILLER_SUFFIX,
    INSTRUCTION,
    OVERLAPS,
    STUDY_COUNT,
    TEST_CUE,
    filler_text,
)
from obm_task2.design import Obj, iter_trials, overlap_degree


FORBIDDEN_PROMPT_STRINGS = [
    "critical_recombination", "near_lure", "unrelated_lure", "probe_type",
    "answer", "repetition", "overlap", "target",
    *EXTRA_COLORS, *EXTRA_SHAPES, *EXTRA_TEXTURES,
]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(root: Path) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    checked_images: set[str] = set()
    labels = _read_jsonl(root / "labels" / "labels.jsonl")
    manifest = _read_jsonl(root / "manifest.jsonl")
    expected = {trial.trial_id: trial for trial in iter_trials()}
    check(len(labels) == len(expected) == len(manifest), "trial count does not match the generator")
    label_ids = [row["trial_id"] for row in labels]
    check(len(label_ids) == len(set(label_ids)), "duplicate trial_id in labels")

    by_match = defaultdict(list)
    for row in labels:
        trial = expected.get(row["trial_id"])
        check(trial is not None, f"unexpected label {row['trial_id']}")
        if trial is None:
            continue
        check(row["answer"] == trial.answer, f"{row['trial_id']}: answer")
        check(row["probe_type"] == trial.probe_type, f"{row['trial_id']}: probe type")
        check(row["repetition"] == trial.repetition, f"{row['trial_id']}: repetition")
        check(row["overlap"] == trial.overlap, f"{row['trial_id']}: overlap")
        check(row["context_newlines"] == trial.context_newlines, f"{row['trial_id']}: newlines")
        check((row["split"] == "dev") == (row["seed"] <= 6), f"{row['trial_id']}: split")
        study = [Obj(**{key: item[key] for key in ("color", "shape", "texture")}) for item in row["study_objects"]]
        probe = Obj(**{key: row["probe_object"][key] for key in ("color", "shape", "texture")})
        distractors = [Obj(**{key: item[key] for key in ("color", "shape", "texture")}) for item in row["distractor_objects"]]
        check(len(study) == STUDY_COUNT and len(distractors) == DISTRACTOR_COUNT, f"{row['trial_id']}: image counts")
        check(all(overlap_degree(item, study) == row["overlap"] for item in distractors), f"{row['trial_id']}: overlap degree")
        colors = [item.color for item in study]
        if row["repetition"] == 1:
            check(len(set(colors)) == STUDY_COUNT, f"{row['trial_id']}: repetition 1")
        elif row["repetition"] == 2:
            check(sorted(colors.count(color) for color in set(colors)) == [2, 2, 2], f"{row['trial_id']}: repetition 2")
        else:
            check(
                sorted(colors.count(color) for color in set(colors)) == [1, 2, 3],
                f"{row['trial_id']}: repetition 3",
            )
        prompt_path = root / row["prompt_file"]
        check(prompt_path.is_file(), f"{row['trial_id']}: missing prompt")
        if not prompt_path.is_file():
            continue
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        check(set(prompt) == {"trial_id", "image_paths", "messages"}, f"{row['trial_id']}: prompt fields")
        texts = [item["text"] for item in prompt["messages"] if item["type"] == "text"]
        images = [item["path"] for item in prompt["messages"] if item["type"] == "image"]
        check(texts[0] == INSTRUCTION and texts[2] == DISTRACTOR_CUE and texts[3] == TEST_CUE, f"{row['trial_id']}: cues")
        expected_filler = filler_text(
            row["context_newlines"],
            seed=row["seed"],
            repetition=row["repetition"],
            overlap=row["overlap"],
        )
        check(texts[1] == expected_filler, f"{row['trial_id']}: filler block")
        check(
            texts[1].startswith(FILLER_PREFIX) and texts[1].endswith(FILLER_SUFFIX),
            f"{row['trial_id']}: filler anchors",
        )
        inner = texts[1][len(FILLER_PREFIX) : -len(FILLER_SUFFIX)]
        check(inner.startswith("\n") and inner.endswith("\n"), f"{row['trial_id']}: filler newlines")
        digit_body = inner[1:-1]
        check(
            len(digit_body) == row["context_newlines"]
            and (digit_body == "" or digit_body.isdigit()),
            f"{row['trial_id']}: digit count",
        )
        check(images == row["study_images"] + row["distractor_images"] + [row["test_image"]], f"{row['trial_id']}: image order")
        check(prompt["image_paths"] == images, f"{row['trial_id']}: image_paths")
        blob = json.dumps(prompt, ensure_ascii=False)
        for banned in FORBIDDEN_PROMPT_STRINGS:
            if banned in blob:
                errors.append(f"{row['trial_id']}: prompt contains {banned}")
                break
        for relative in images:
            path = root / relative
            if not path.is_file():
                errors.append(f"{row['trial_id']}: missing {relative}")
                continue
            if relative not in checked_images:
                with Image.open(path) as image:
                    if image.size != (384, 384) or image.format != "PNG":
                        errors.append(f"{relative}: not a 384 PNG")
                checked_images.add(relative)
        by_match[(row["seed"], row["repetition"], row["probe_index"])].append(row)

    for key, rows in by_match.items():
        check(len(rows) == len(OVERLAPS) * len(CONTEXT_NEWLINES), f"{key}: missing crossed conditions")
        study_images = {tuple(row["study_images"]) for row in rows}
        probes = {row["test_image"] for row in rows}
        answers = {row["answer"] for row in rows}
        check(len(study_images) == 1 and len(probes) == 1 and len(answers) == 1, f"{key}: study or probe changed")
        for overlap in OVERLAPS:
            matched = [row for row in rows if row["overlap"] == overlap]
            distractors = {tuple(row["distractor_images"]) for row in matched}
            check(len(distractors) == 1, f"{key} overlap {overlap}: distractors changed with newlines")
    return errors
