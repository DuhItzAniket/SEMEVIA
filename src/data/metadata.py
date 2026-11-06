"""Class metadata and the make -> family -> category hierarchy.

Make names are matched against an explicit list (never parsed from the first word), so
multiword makes such as "AM General", "Aston Martin" and "Land Rover" are handled.

Family rule (documented application grouping, metadata_version below):
  family = the class name with only its trailing model year removed.
  Categories therefore share a family only when make, model, trim and body style are
  identical and they differ solely by year (e.g. Audi S4 Sedan 2012 / 2007). Body-style
  and trim distinctions (S5 Coupe vs S5 Convertible) are never merged. This is the most
  conservative grouping; the full 196-row table is written to artifacts for review.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

METADATA_VERSION = "meta_v1_family_eq_name_minus_year"

MULTIWORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


@dataclass(frozen=True)
class ClassInfo:
    class_id: int            # 0..195
    original_class_id: int   # 1..196
    name: str
    make: str
    family: str
    model: str               # name without make and year (model, trim, body)
    body_style: str | None
    year: int

    def to_dict(self) -> dict:
        return asdict(self)


BODY_STYLES = (
    "Sedan", "Coupe", "Convertible", "Hatchback", "SUV", "Wagon", "Minivan", "Van",
    "Cab", "Roadster", "Cabriolet",
)


def _body_style(model: str) -> str | None:
    tokens = model.split()
    for tok in reversed(tokens):
        if tok in BODY_STYLES:
            return tok
    return None


def parse_class_name(name: str, class_id: int) -> ClassInfo:
    name = " ".join(name.split())
    m = re.match(r"^(.*)\s(\d{4})$", name)
    if not m:
        raise ValueError(f"class name without trailing year: {name!r}")
    stem, year = m.group(1), int(m.group(2))
    make = next((mk for mk in MULTIWORD_MAKES if stem.startswith(mk + " ")), stem.split(" ", 1)[0])
    model = stem[len(make):].strip()
    if not model:
        raise ValueError(f"empty model for {name!r}")
    return ClassInfo(
        class_id=class_id,
        original_class_id=class_id + 1,
        name=name,
        make=make,
        family=stem,
        model=model,
        body_style=_body_style(model),
        year=year,
    )


def build_metadata(class_names: list[str]) -> list[ClassInfo]:
    if len(class_names) != 196:
        raise ValueError(f"expected 196 class names, got {len(class_names)}")
    infos = [parse_class_name(n, i) for i, n in enumerate(class_names)]
    if len({i.name for i in infos}) != 196:
        raise ValueError("duplicate class names")
    return infos
