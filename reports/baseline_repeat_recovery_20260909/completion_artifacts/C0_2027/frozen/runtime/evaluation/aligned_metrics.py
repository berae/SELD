"""Common, explicitly versioned DCASE scoring for EINV2 and Multi-ACCDOA.

No model-specific metric code. Upstream files are immutable and SHA256-pinned.
The 2020 source returns localization F, NOT the webpage's localization recall.
"""
import csv
import hashlib
import math
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parent
UPSTREAM = {
    "2020": ("384ad9baa634e91268f35759fd7382e0d2f0ef12",
             "a6b4021b14b1e98d3c7014914c50c8d5ffcd263159d45b070703c64d1bb61ceb"),
    "2023": ("24d14c9ebe6878062dc3b68eb07dd0100722f530",
             "41b17ffbd5007cd06549b07ad620a77d7cdbc65ae62c93e409f37f28882512c9"),
}


def load_official(year):
    """Load pinned official arithmetic, with only NumPy/import compatibility."""
    path = ROOT / "upstream" / ("dcase%s_metrics.py" % year)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM[year][1]:
        raise ValueError("Upstream SHA256 mismatch: %s" % path)
    source = raw.decode("utf-8")
    # Do not monkey-patch numpy globally or change any metric arithmetic.
    source = source.replace("np.finfo(np.float)", "np.finfo(float)")
    source = source.replace("from IPython import  embed", "")
    source = source.replace("from IPython import embed", "")
    module = ModuleType("official_dcase" + year)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


OFFICIAL_2020 = load_official("2020")
OFFICIAL_2023 = load_official("2023")
SCHEMAS = {
    "polar4": (4, False, False),
    "polar5": (5, True, False),
    "polar6": (6, True, False),  # final distance column unused
    "cartesian5": (5, False, True),
    "cartesian6": (6, True, True),
    "cartesian7": (7, True, True),  # final distance column unused
}


def load_csv(path, schema, frames=600, classes=14):
    """Return frame -> [class, track, azimuth_degrees, elevation_degrees].

    Schemas are explicit: five/six columns are otherwise ambiguous across years.
    Preserve CSV source order and duplicate same-class sources (no deduplication).
    """
    columns, has_track, cartesian = SCHEMAS[schema]
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line, values in enumerate(csv.reader(stream), 1):
            if not values:
                continue
            if len(values) != columns:
                raise ValueError("%s:%d: expected %s" % (path, line, schema))
            frame, cls = int(values[0]), int(values[1])
            if not 0 <= frame < frames or not 0 <= cls < classes:
                raise ValueError("%s:%d: frame/class out of range" % (path, line))
            track = int(values[2]) if has_track else 0
            start = 3 if has_track else 2
            coords = [float(x) for x in values[start:start + (3 if cartesian else 2)]]
            if not all(math.isfinite(x) for x in coords):
                raise ValueError("%s:%d: non-finite direction" % (path, line))
            if cartesian:
                x, y, z = coords
                if x * x + y * y + z * z == 0:
                    raise ValueError("%s:%d: zero direction vector" % (path, line))
                az = float(np.arctan2(y, x) * 180 / np.pi)
                el = float(np.arctan2(z, np.sqrt(x*x + y*y)) * 180 / np.pi)
            else:
                az, el = coords
            labels.setdefault(frame, []).append([cls, track, az, el])
    return labels


def segment_labels(labels, frames=600, frames_per_second=10, keep_track=True):
    """Official one-second block layout; cover the FULL recording duration.

    2020 expects [az, el], 2023 expects [track, az, el] within each frame.
    The official metric performs its own matching; no ground-truth-based filtering.
    """
    blocks = {i: {} for i in range(int(math.ceil(frames / frames_per_second)))}
    for start in range(0, frames, frames_per_second):
        grouped = {}
        for frame in range(start, min(start + frames_per_second, frames)):
            for value in labels.get(frame, []):
                grouped.setdefault(value[0], {}).setdefault(frame - start, []).append(
                    value[1:] if keep_track else value[2:])
        for cls, per_frame in grouped.items():
            blocks[start // frames_per_second][cls] = [[list(per_frame), list(per_frame.values())]]
    return blocks


class AlignedMetrics:
    """Score identical framewise labels using two frozen official protocols."""
    def __init__(self, classes=14, threshold=20, frames=600, frames_per_second=10):
        self.legacy = OFFICIAL_2020.SELDMetrics(nb_classes=classes, doa_threshold=threshold)
        self.modern = OFFICIAL_2023.SELDMetrics(nb_classes=classes, doa_threshold=threshold, average="micro")
        self.frames = frames
        self.fps = frames_per_second

    def update(self, pred, gt):
        old_pred = segment_labels(pred, self.frames, self.fps, False)
        old_gt = segment_labels(gt, self.frames, self.fps, False)
        new_pred = segment_labels(pred, self.frames, self.fps, True)
        new_gt = segment_labels(gt, self.frames, self.fps, True)
        self.legacy.update_seld_scores(old_pred, old_gt)
        self.modern.update_seld_scores(new_pred, new_gt)

    def scores(self):
        er, f, le, lf = self.legacy.compute_seld_scores()
        # Separate, explicitly adapted profile: webpage LR definition + 2020 counts.
        # This is NOT claimed to be the unmodified official 2020 code output.
        lr = self.legacy._DE_TP / (self.legacy._Nref + OFFICIAL_2020.eps)
        result = {
            "dcase2020_source": dict(ER20=er, F20=f, LE_CD=le, LF_CD=lf,
                                     SELD_LF=OFFICIAL_2020.early_stopping_metric([er, f], [le, lf])),
            "dcase2020_web_lr": dict(ER20=er, F20=f, LE_CD=le, LR_CD=lr,
                                     SELD_LR=OFFICIAL_2020.early_stopping_metric([er, f], [le, lr])),
        }
        for average in ("micro", "macro"):
            self.modern._average = average
            er, f, le, lr, seld, _ = self.modern.compute_seld_scores()
            result["dcase2023_" + average] = dict(ER20=er, F20=f, LE_CD=le, LR_CD=lr, SELD_LR=seld)
        self.modern._average = "micro"
        return {profile: {key: float(value) for key, value in values.items()}
                for profile, values in result.items()}

    def counters(self):
        old_keys = ("_Nref", "_Nsys", "_DE_TP", "_total_DE", "_TP", "_FP", "_FN", "_S", "_D", "_I")
        new_keys = ("_Nref", "_DE_TP", "_DE_FP", "_DE_FN", "_total_DE", "_TP", "_FP", "_FP_spatial", "_FN", "_S", "_D", "_I")
        return {year: {key: np.asarray(getattr(obj, key)).tolist() for key in keys}
                for year, obj, keys in (("2020", self.legacy, old_keys), ("2023", self.modern, new_keys))}
