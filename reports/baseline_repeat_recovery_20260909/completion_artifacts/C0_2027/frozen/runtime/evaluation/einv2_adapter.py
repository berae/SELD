"""EINV2's four-value API backed by the pinned DCASE2023 micro scorer.

The old 2020 module remains available for historical reproduction. The aligned
consumer imports this adapter explicitly, so LR20 now really is recall.
"""
from aligned_metrics import OFFICIAL_2020, OFFICIAL_2023

METRIC_PROTOCOL = "dcase2023_micro_full_duration"
early_stopping_metric = OFFICIAL_2020.early_stopping_metric


def add_track_column(blocks):
    """2020 [az,el]/[x,y,z] layout -> 2023 [track,az,el]/[track,x,y,z]."""
    converted = {}
    for block, classes in blocks.items():
        converted[block] = {}
        for cls, entries in classes.items():
            converted[block][cls] = []
            for frames, frame_values in entries:
                converted[block][cls].append([
                    frames, [[[track] + list(coords) for track, coords in enumerate(values)]
                             for values in frame_values]])
    return converted


class SELDMetrics:
    metric_protocol = METRIC_PROTOCOL

    def __init__(self, doa_threshold=20, nb_classes=11):
        self.scorer = OFFICIAL_2023.SELDMetrics(doa_threshold=doa_threshold,
                                             nb_classes=nb_classes, average="micro")

    def update_seld_scores(self, pred, gt):
        self.scorer.update_seld_scores(add_track_column(pred), add_track_column(gt))

    update_seld_scores_xyz = update_seld_scores

    def compute_seld_scores(self):
        return self.scorer.compute_seld_scores()[:4]
