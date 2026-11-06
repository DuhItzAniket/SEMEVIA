"""Root -> make -> family -> category hierarchy and deepest-common-ancestor specificity.

Built from artifacts/class_metadata.json (metadata_version recorded). A family node exists only
for multi-member families (categories that differ solely by year); single-member families
collapse onto the category, so a candidate set of one category answers at category level.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Specificity:
    level: str          # "category" | "family" | "make" | "root" | "empty" | "full_set"
    label: str | None   # human-readable node name, None for root/empty
    unresolved: list[str]  # what remains open (e.g. category years)

    def to_dict(self) -> dict:
        return {"level": self.level, "label": self.label, "unresolved": self.unresolved}


class Hierarchy:
    def __init__(self, classes: list[dict], metadata_version: str):
        self.version = metadata_version
        self.classes = {c["class_id"]: c for c in classes}
        if sorted(self.classes) != list(range(len(classes))):
            raise ValueError("class ids must be 0..n-1")
        fam_sizes: dict[str, int] = {}
        for c in classes:
            fam_sizes[c["family"]] = fam_sizes.get(c["family"], 0) + 1
        self.multi_families = {f for f, n in fam_sizes.items() if n > 1}

    @classmethod
    def load(cls, path: Path = ROOT / "artifacts" / "class_metadata.json") -> "Hierarchy":
        meta = json.loads(path.read_text(encoding="utf-8"))
        return cls(meta["classes"], meta["metadata_version"])

    def path(self, class_id: int) -> tuple[str, str | None, int]:
        c = self.classes[class_id]
        fam = c["family"] if c["family"] in self.multi_families else None
        return c["make"], fam, class_id

    def specificity(self, candidate_ids: list[int], is_failure: bool = False) -> Specificity:
        n_all = len(self.classes)
        if is_failure:
            return Specificity("full_set", None, ["no usable detection or prediction"])
        if not candidate_ids:
            return Specificity("empty", None, ["no category met the calibrated threshold"])
        ids = sorted(set(candidate_ids))
        if len(ids) == n_all:
            return Specificity("root", None, ["all categories remain"])
        if len(ids) == 1:
            return Specificity("category", self.classes[ids[0]]["name"], [])
        paths = [self.path(i) for i in ids]
        makes = {p[0] for p in paths}
        if len(makes) > 1:
            return Specificity("root", None, [f"{len(makes)} makes remain"])
        fams = {p[1] for p in paths}
        if len(fams) == 1 and None not in fams:
            years = sorted({self.classes[i]["year"] for i in ids})
            return Specificity("family", fams.pop(), [f"category year unresolved: {', '.join(map(str, years))}"])
        names = [self.classes[i]["name"] for i in ids]
        return Specificity("make", makes.pop(), [f"{len(ids)} categories remain: " + "; ".join(names[:6]) + ("..." if len(names) > 6 else "")])

    def is_specific(self, s: Specificity) -> bool:
        """Root, empty and failure answers count as insufficient evidence."""
        return s.level in ("category", "family", "make")
