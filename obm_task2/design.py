"""Build the Task 2 factorial without writing images.

Three factors are crossed. Study length and distractor count do not change.

- repetition 1: six study objects, each color, shape, and texture used once
- repetition 2: six study objects from three colors, shapes, and textures,
  each used twice
- repetition 3: six study objects from three colors, shapes, and textures,
  used 3, 2, and 1 times respectively (max within-list repetition is 3)
- overlap 0: three distractors whose attributes are all outside the study set
- overlap 1: three distractors, sharing color, shape, and texture respectively
- overlap 2: three distractors, each sharing exactly two attributes with the study set
- context_newlines: digit counts between two newlines (0, 1024, …, 16384)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

from obm_task2.constants import (
    COLORS,
    CONTEXT_NEWLINES,
    DEV_SEEDS,
    DISTRACTOR_COUNT,
    EXTRA_COLORS,
    EXTRA_SHAPES,
    EXTRA_TEXTURES,
    NEAR_COLOR,
    OVERLAPS,
    REPETITIONS,
    SEEDS,
    SHAPES,
    STUDY_COUNT,
    TEXTURES,
    UNRELATED,
    object_zh,
)


@dataclass(frozen=True)
class Obj:
    color: str
    shape: str
    texture: str

    def key(self) -> str:
        return f"{self.color}|{self.shape}|{self.texture}"

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "shape": self.shape,
            "texture": self.texture,
            "zh": object_zh(self.color, self.shape, self.texture),
        }


# Each index 0, 1, 2 occurs twice. Three fully unbound triples remain.
REP2_TEMPLATE = ((0, 0, 0), (0, 1, 1), (1, 1, 2), (1, 2, 0), (2, 2, 1), (2, 0, 2))
REP2_CRITICAL = ((0, 2, 2), (1, 0, 1), (2, 1, 0))
REP2_NEAR_PAIRS = ((0, 1), (1, 0), (2, 2))
UNRELATED_OBJECT = Obj(*UNRELATED)


@dataclass
class Trial:
    trial_id: str
    group_id: str
    split: str
    seed: int
    repetition: int
    overlap: int
    context_newlines: int
    probe_index: int
    probe_type: str
    answer: str
    study_objects: list[Obj]
    probe_object: Obj
    distractor_objects: list[Obj]
    probe_id: str

    def label_row(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "group_id": self.group_id,
            "split": self.split,
            "seed": self.seed,
            "repetition": self.repetition,
            "overlap": self.overlap,
            "context_newlines": self.context_newlines,
            "probe_index": self.probe_index,
            "probe_type": self.probe_type,
            "answer": self.answer,
            "probe_id": self.probe_id,
            "study_objects": [item.as_dict() for item in self.study_objects],
            "probe_object": self.probe_object.as_dict(),
            "distractor_objects": [item.as_dict() for item in self.distractor_objects],
        }


def _seen(study: list[Obj]) -> tuple[set[str], set[str], set[str]]:
    return (
        {item.color for item in study},
        {item.shape for item in study},
        {item.texture for item in study},
    )


def overlap_degree(obj: Obj, study: list[Obj]) -> int:
    colors, shapes, textures = _seen(study)
    return (obj.color in colors) + (obj.shape in shapes) + (obj.texture in textures)


def pairs_absent(obj: Obj, study: list[Obj]) -> bool:
    for item in study:
        if item.color == obj.color and item.shape == obj.shape:
            return False
        if item.color == obj.color and item.texture == obj.texture:
            return False
        if item.shape == obj.shape and item.texture == obj.texture:
            return False
    return True


def _assert_repetition(study: list[Obj], repetition: int) -> None:
    colors, shapes, textures = _seen(study)
    if any(item.color in EXTRA_COLORS or item.shape in EXTRA_SHAPES or item.texture in EXTRA_TEXTURES for item in study):
        raise RuntimeError("study list used a held-out attribute")
    if repetition == 1:
        if not (len(colors) == len(shapes) == len(textures) == STUDY_COUNT):
            raise RuntimeError("repetition 1 is not one occurrence per attribute")
        return
    if repetition == 2:
        if not (len(colors) == len(shapes) == len(textures) == 3):
            raise RuntimeError("repetition 2 does not use three values per attribute")
        for values, getter in ((colors, lambda item: item.color), (shapes, lambda item: item.shape), (textures, lambda item: item.texture)):
            for value in values:
                if sum(getter(item) == value for item in study) != 2:
                    raise RuntimeError("repetition 2 is not exactly two occurrences")
        return
    if repetition == 3:
        if not (len(colors) == len(shapes) == len(textures) == 3):
            raise RuntimeError("repetition 3 does not use three values per attribute")
        for values, getter in ((colors, lambda item: item.color), (shapes, lambda item: item.shape), (textures, lambda item: item.texture)):
            if sorted(sum(getter(item) == value for item in study) for value in values) != [1, 2, 3]:
                raise RuntimeError("repetition 3 is not counts 3, 2, 1 per attribute")
        return
    raise RuntimeError(f"unknown repetition {repetition}")


def _assert_probe(kind: str, obj: Obj, study: list[Obj]) -> None:
    colors, shapes, textures = _seen(study)
    if kind == "target":
        if obj not in study:
            raise RuntimeError("target is not a study object")
        return
    if obj in study:
        raise RuntimeError(f"{kind} collides with a study object")
    if kind == "critical_recombination":
        if obj.color not in colors or obj.shape not in shapes or obj.texture not in textures:
            raise RuntimeError("critical probe uses an unseen attribute")
        if not pairs_absent(obj, study):
            raise RuntimeError("critical probe reuses a studied pair")
        return
    if kind == "near_lure":
        if obj.color != NEAR_COLOR or obj.color in colors:
            raise RuntimeError("near lure color is not the held-out color")
        if obj.shape not in shapes or obj.texture not in textures:
            raise RuntimeError("near lure shape or texture was not studied")
        if any(item.shape == obj.shape and item.texture == obj.texture for item in study):
            raise RuntimeError("near lure reuses a studied shape-texture pair")
        return
    if kind == "unrelated_lure":
        if obj != UNRELATED_OBJECT:
            raise RuntimeError("unexpected unrelated object")
        if overlap_degree(obj, study) != 0:
            raise RuntimeError("unrelated lure shares a study attribute")
        return
    raise RuntimeError(kind)


def _assign_321(picker: random.Random, values: list[str]) -> tuple[list[str], list[str]]:
    """Three sampled values with multiplicities 3, 2, 1 (shuffled)."""
    chosen = picker.sample(values, 3)
    slots = [chosen[0]] * 3 + [chosen[1]] * 2 + [chosen[2]]
    picker.shuffle(slots)
    return chosen, slots


def _study_list(seed: int, repetition: int) -> tuple[list[Obj], list[Obj], list[Obj]]:
    picker = random.Random(f"study-{seed}-{repetition}")
    if repetition == 1:
        colors = picker.sample(list(COLORS), STUDY_COUNT)
        shapes = picker.sample(SHAPES, STUDY_COUNT)
        textures = picker.sample(TEXTURES, STUDY_COUNT)
        study = [Obj(colors[i], shapes[i], textures[i]) for i in range(STUDY_COUNT)]
        critical_pool = [
            Obj(colors[a], shapes[b], textures[c])
            for a, b, c in product(range(STUDY_COUNT), repeat=3)
            if len({a, b, c}) == 3
        ]
        near_pool = [
            Obj(NEAR_COLOR, shapes[a], textures[b])
            for a, b in product(range(STUDY_COUNT), repeat=2)
            if a != b
        ]
    elif repetition == 2:
        colors = picker.sample(list(COLORS), 3)
        shapes = picker.sample(SHAPES, 3)
        textures = picker.sample(TEXTURES, 3)
        study = [Obj(colors[i], shapes[j], textures[k]) for i, j, k in REP2_TEMPLATE]
        critical_pool = [Obj(colors[i], shapes[j], textures[k]) for i, j, k in REP2_CRITICAL]
        near_pool = [Obj(NEAR_COLOR, shapes[j], textures[k]) for j, k in REP2_NEAR_PAIRS]
    else:
        for attempt in range(5000):
            trial_rng = random.Random(f"study-{seed}-{repetition}-{attempt}")
            _, color_slots = _assign_321(trial_rng, list(COLORS))
            _, shape_slots = _assign_321(trial_rng, list(SHAPES))
            _, texture_slots = _assign_321(trial_rng, list(TEXTURES))
            study = [
                Obj(color_slots[i], shape_slots[i], texture_slots[i])
                for i in range(STUDY_COUNT)
            ]
            if len({item.key() for item in study}) != STUDY_COUNT:
                continue
            try:
                _assert_repetition(study, repetition)
            except RuntimeError:
                continue
            colors = sorted({item.color for item in study})
            shapes = sorted({item.shape for item in study})
            textures = sorted({item.texture for item in study})
            color_i = {c: i for i, c in enumerate(colors)}
            shape_i = {s: i for i, s in enumerate(shapes)}
            texture_i = {t: i for i, t in enumerate(textures)}
            critical_pool = [
                candidate
                for a, b, c in product(range(3), repeat=3)
                if len({a, b, c}) == 3
                for candidate in [Obj(colors[a], shapes[b], textures[c])]
                if candidate not in study and pairs_absent(candidate, study)
            ]
            near_pool = [
                candidate
                for j, k in product(range(3), repeat=2)
                if j != k
                for candidate in [Obj(NEAR_COLOR, shapes[j], textures[k])]
                if candidate not in study
                and not any(item.shape == candidate.shape and item.texture == candidate.texture for item in study)
            ]
            if len(critical_pool) < 3 or not near_pool:
                continue
            break
        else:
            raise RuntimeError(f"could not build repetition-3 study for seed {seed}")
    order = random.Random(f"order-{seed}-{repetition}")
    order.shuffle(study)
    _assert_repetition(study, repetition)
    if len(critical_pool) < 3 or not near_pool:
        raise RuntimeError(f"not enough lures for seed {seed} repetition {repetition}")
    return study, critical_pool, near_pool


def _probes(seed: int, repetition: int, study: list[Obj], critical_pool: list[Obj], near_pool: list[Obj]):
    rng = random.Random(f"probes-{seed}-{repetition}")
    targets = rng.sample(study, 3)
    criticals = rng.sample(critical_pool, 3)
    near = rng.choice(near_pool)
    probes = (
        [("target", "YES", item) for item in targets]
        + [("critical_recombination", "NO", item) for item in criticals]
        + [("near_lure", "NO", near), ("unrelated_lure", "NO", UNRELATED_OBJECT)]
    )
    rng.shuffle(probes)
    for kind, _answer, obj in probes:
        _assert_probe(kind, obj, study)
    return probes


def _distractors(seed: int, repetition: int, overlap: int, study: list[Obj], probes: list[tuple]) -> list[Obj]:
    blocked = {obj for _kind, _answer, obj in probes}
    rng = random.Random(f"distractor-{seed}-{repetition}-{overlap}")
    if overlap == 0:
        pool = [
            Obj(color, shape, texture)
            for color, shape, texture in product(EXTRA_COLORS, EXTRA_SHAPES, EXTRA_TEXTURES)
            if Obj(color, shape, texture) not in blocked
        ]
        rng.shuffle(pool)
        chosen = pool[:DISTRACTOR_COUNT]
    elif overlap == 1:
        colors, shapes, textures = _seen(study)
        shared = {
            "color": rng.sample(sorted(colors), 1)[0],
            "shape": rng.sample(sorted(shapes), 1)[0],
            "texture": rng.sample(sorted(textures), 1)[0],
        }
        # One distractor for each attribute. The other two attributes are held out,
        # so the shared count cannot accidentally become two.
        candidates = {
            "color": [Obj(shared["color"], shape, texture) for shape, texture in product(EXTRA_SHAPES, EXTRA_TEXTURES)],
            "shape": [Obj(color, shared["shape"], texture) for color, texture in product(EXTRA_COLORS, EXTRA_TEXTURES)],
            "texture": [Obj(color, shape, shared["texture"]) for color, shape in product(EXTRA_COLORS, EXTRA_SHAPES)],
        }
        chosen = []
        for role in ("color", "shape", "texture"):
            options = [item for item in candidates[role] if item not in blocked and item not in chosen]
            rng.shuffle(options)
            if not options:
                raise RuntimeError(f"no overlap-1 distractor for {role}")
            chosen.append(options[0])
    elif overlap == 2:
        colors, shapes, textures = _seen(study)
        candidates = {
            "color_shape": [
                Obj(color, shape, texture)
                for color in colors
                for shape in shapes
                for texture in EXTRA_TEXTURES
            ],
            "color_texture": [
                Obj(color, shape, texture)
                for color in colors
                for texture in textures
                for shape in EXTRA_SHAPES
            ],
            "shape_texture": [
                Obj(color, shape, texture)
                for shape in shapes
                for texture in textures
                for color in EXTRA_COLORS
            ],
        }
        chosen = []
        for role in ("color_shape", "color_texture", "shape_texture"):
            options = [
                item
                for item in candidates[role]
                if overlap_degree(item, study) == 2
                and item not in blocked
                and item not in chosen
            ]
            rng.shuffle(options)
            if not options:
                raise RuntimeError(f"no overlap-2 distractor for {role}")
            chosen.append(options[0])
    else:
        raise RuntimeError(f"unknown overlap {overlap}")
    if len(chosen) != DISTRACTOR_COUNT or len(set(chosen)) != DISTRACTOR_COUNT:
        raise RuntimeError("distractor list is not three distinct objects")
    for item in chosen:
        if overlap_degree(item, study) != overlap:
            raise RuntimeError(f"distractor overlap is {overlap_degree(item, study)}, expected {overlap}")
        if item in blocked:
            raise RuntimeError("distractor matches a probe")
    return chosen


def iter_trials() -> list[Trial]:
    trials: list[Trial] = []
    for seed in SEEDS:
        for repetition in REPETITIONS:
            study, critical_pool, near_pool = _study_list(seed, repetition)
            probes = _probes(seed, repetition, study, critical_pool, near_pool)
            for overlap in OVERLAPS:
                distractors = _distractors(seed, repetition, overlap, study, probes)
                for newlines in CONTEXT_NEWLINES:
                    group_id = f"s{seed:02d}_rep{repetition}_ov{overlap}_nl{newlines}"
                    for probe_index, (kind, answer, obj) in enumerate(probes, start=1):
                        trials.append(Trial(
                            trial_id=f"{group_id}_t{probe_index}",
                            group_id=group_id,
                            split="dev" if seed in DEV_SEEDS else "test",
                            seed=seed,
                            repetition=repetition,
                            overlap=overlap,
                            context_newlines=newlines,
                            probe_index=probe_index,
                            probe_type=kind,
                            answer=answer,
                            study_objects=list(study),
                            probe_object=obj,
                            distractor_objects=list(distractors),
                            probe_id=f"s{seed:02d}_rep{repetition}_p{probe_index}",
                        ))
    return trials
