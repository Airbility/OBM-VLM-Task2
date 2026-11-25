"""Load Task 2 prompts without exposing the answer key."""

from __future__ import annotations

import json
from pathlib import Path


class Task2Dataset:
    """Iterate model inputs. Labels stay behind ``get_label``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = [
            json.loads(line)
            for line in (self.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._labels = None

    def iter_examples(self, split: str | None = None, repetition: int | None = None,
                      overlap: int | None = None, context_newlines: int | None = None):
        for row in self.manifest:
            if split is not None and row["split"] != split:
                continue
            if repetition is not None and row["repetition"] != repetition:
                continue
            if overlap is not None and row["overlap"] != overlap:
                continue
            if context_newlines is not None and row["context_newlines"] != context_newlines:
                continue
            yield {
                "trial_id": row["trial_id"],
                "split": row["split"],
                "repetition": row["repetition"],
                "overlap": row["overlap"],
                "context_newlines": row["context_newlines"],
                "probe_index": row["probe_index"],
                "messages": self.model_messages(row["trial_id"]),
            }

    def model_messages(self, trial_id: str) -> list[dict]:
        """Text plus absolute image paths. This is the only payload to send."""
        prompt = json.loads((self.root / "prompts" / f"{trial_id}.json").read_text(encoding="utf-8"))
        messages = []
        for item in prompt["messages"]:
            if item["type"] == "text":
                messages.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                messages.append({"type": "image", "path": str((self.root / item["path"]).resolve())})
            else:
                raise ValueError(f"unknown message type in {trial_id}")
        return messages

    def qwen_content(self, trial_id: str) -> list[dict]:
        """Content list accepted by the original Qwen runner."""
        content = []
        for item in self.model_messages(trial_id):
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            else:
                content.append({"type": "image", "image": item["path"]})
        return content

    def get_label(self, trial_id: str) -> dict:
        if self._labels is None:
            self._labels = {
                json.loads(line)["trial_id"]: json.loads(line)
                for line in (self.root / "labels" / "labels.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        return self._labels[trial_id]


def parse_yes_no(text: str | None) -> str | None:
    if not text:
        return None
    value = text.strip().upper()
    return value if value in {"YES", "NO"} else None
