"""Validate an existing Task 2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from obm_task2.validate import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "Dataset")
    args = parser.parse_args()
    errors = validate_dataset(args.dataset)
    if errors:
        print("\n".join(errors[:30]))
        raise SystemExit(f"{len(errors)} errors")
    print("validation: ok")


if __name__ == "__main__":
    main()
