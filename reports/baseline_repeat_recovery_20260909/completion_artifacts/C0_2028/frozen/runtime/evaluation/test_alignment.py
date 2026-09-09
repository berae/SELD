"""Regression tests against frozen upstream source and synthetic edge cases."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from scipy import stats

from aligned_metrics import (AlignedMetrics, OFFICIAL_2020, OFFICIAL_2023, ROOT,
                             UPSTREAM, load_csv, segment_labels)
from einv2_adapter import SELDMetrics as EINV2Adapter


def upstream_segment(year, labels, frames=600):
    tree = ast.parse((ROOT / "upstream" / ("dcase%s_feature.py" % year)).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FeatureClass")
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "segment_labels"))
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), "official_segment_labels", "exec"), namespace)
    state = type("OfficialFeatureState", (), {"_nb_label_frames_1s": 10})()
    return namespace["segment_labels"](state, labels, frames)


def score(pred, gt, classes=3):
    metric = AlignedMetrics(classes=classes)
    metric.update(pred, gt)
    return metric.scores()


class AlignmentTests(unittest.TestCase):
    def test_upstream_hashes(self):
        for year, (_, expected) in UPSTREAM.items():
            self.assertEqual(hashlib.sha256((ROOT / "upstream" / ("dcase%s_metrics.py" % year)).read_bytes()).hexdigest(), expected)

    def test_einv2_legacy_matches_2020(self):
        current = (ROOT / "snapshots/einv2_legacy.py").read_text(encoding="utf-8")
        source = (ROOT / "upstream/dcase2020_metrics.py").read_text(encoding="utf-8").replace("np.finfo(np.float)", "np.finfo(float)")
        self.assertEqual(current, source)

    def test_multi_legacy_matches_2023(self):
        self.assertEqual((ROOT / "snapshots/multi_legacy.py").read_bytes(), (ROOT / "upstream/dcase2023_metrics.py").read_bytes())

    def test_segment_matches_both_official_versions(self):
        labels = {0: [[0, 9, 0., 0.], [0, 3, 90., 0.]], 9: [[1, 2, 30., -4.]],
                  10: [[1, 2, 31., -4.]], 599: [[2, 6, -179., 89.]]}
        old = {frame: [[value[0]] + value[2:] for value in values] for frame, values in labels.items()}
        self.assertEqual(segment_labels(labels, keep_track=False), upstream_segment("2020", old))
        self.assertEqual(segment_labels(labels), upstream_segment("2023", labels))

    def test_source_2020_return_is_lf_not_lr(self):
        s = score({0: [[0, 0, 0., 0.], [1, 0, 0., 0.]]}, {0: [[0, 0, 0., 0.]]})
        self.assertAlmostEqual(s["dcase2020_source"]["LF_CD"], 2/3)
        self.assertNotIn("LR_CD", s["dcase2020_source"])
        self.assertAlmostEqual(s["dcase2020_web_lr"]["LR_CD"], 1.)
        self.assertAlmostEqual(s["dcase2023_micro"]["LR_CD"], 1.)

    def test_missing_reference_reduces_recall(self):
        s = score({0: [[0, 0, 0., 0.]]}, {0: [[0, 0, 0., 0.], [1, 0, 90., 0.]]})
        self.assertAlmostEqual(s["dcase2020_web_lr"]["LR_CD"], .5)
        self.assertAlmostEqual(s["dcase2023_micro"]["LR_CD"], .5)

    def test_multisource_protocol_difference(self):
        s = score({0: [[0, 0, 0., 0.]]}, {0: [[0, 1, 0., 0.], [0, 2, 90., 0.]]})
        self.assertAlmostEqual(s["dcase2020_web_lr"]["LR_CD"], 1.)
        self.assertAlmostEqual(s["dcase2023_micro"]["LR_CD"], .5)

    def test_le_can_change_between_versions(self):
        s = score({0: [[0, 0, 10., 0.], [0, 0, 100., 0.]]}, {0: [[0, 1, 0., 0.], [0, 2, 90., 0.]]})
        self.assertAlmostEqual(s["dcase2020_source"]["LE_CD"], 20.)
        self.assertAlmostEqual(s["dcase2023_micro"]["LE_CD"], 10.)

    def test_bad_angle_not_recall_threshold(self):
        s = score({0: [[0, 0, 30., 0.]]}, {0: [[0, 0, 0., 0.]]})
        for key in ("dcase2020_web_lr", "dcase2023_micro"):
            self.assertAlmostEqual(s[key]["LE_CD"], 30.)
            self.assertAlmostEqual(s[key]["LR_CD"], 1.)
            self.assertAlmostEqual(s[key]["F20"], 0.)

    def test_no_predictions(self):
        s = score({}, {0: [[0, 0, 0., 0.]]})
        self.assertEqual(s["dcase2023_micro"]["LE_CD"], 180.)
        self.assertEqual(s["dcase2023_micro"]["LR_CD"], 0.)
        self.assertAlmostEqual(s["dcase2023_micro"]["ER20"], 1.)

    def test_same_block_without_temporal_overlap(self):
        s = score({9: [[0, 0, 0., 0.]]}, {0: [[0, 0, 0., 0.]]})
        self.assertEqual(s["dcase2020_web_lr"]["LR_CD"], 0.)
        self.assertEqual(s["dcase2023_micro"]["LR_CD"], 0.)

    def test_full_clip_counts_late_false_positive(self):
        gt = {0: [[0, 0, 0., 0.]]}
        pred = {0: [[0, 0, 0., 0.]], 599: [[1, 0, 0., 0.]]}
        s = score(pred, gt)
        self.assertAlmostEqual(s["dcase2023_micro"]["ER20"], 1.)
        self.assertIn(59, segment_labels(pred))
        self.assertIn(1, segment_labels(pred)[59])

    def test_macro_is_not_micro(self):
        s = score({0: [[0, 0, 0., 0.]]}, {0: [[0, 0, 0., 0.]]})
        self.assertAlmostEqual(s["dcase2023_micro"]["LR_CD"], 1.)
        self.assertAlmostEqual(s["dcase2023_macro"]["LR_CD"], 1/3)
        self.assertAlmostEqual(s["dcase2023_macro"]["LE_CD"], 120.)

    def test_accumulator_matches_direct_upstream(self):
        rng = np.random.RandomState(26)
        pred, gt = {}, {}
        for frame in range(35):
            for target in (pred, gt):
                target[frame] = [[int(rng.randint(3)), i, float(rng.uniform(-180, 180)), float(rng.uniform(-70, 70))]
                                 for i in range(int(rng.randint(4)))]
        ours = score(pred, gt)
        old = OFFICIAL_2020.SELDMetrics(nb_classes=3)
        modern = OFFICIAL_2023.SELDMetrics(nb_classes=3, average="micro")
        old_pred = {t: [[v[0]] + v[2:] for v in values] for t, values in pred.items()}
        old_gt = {t: [[v[0]] + v[2:] for v in values] for t, values in gt.items()}
        old.update_seld_scores(upstream_segment("2020", old_pred), upstream_segment("2020", old_gt))
        modern.update_seld_scores(upstream_segment("2023", pred), upstream_segment("2023", gt))
        np.testing.assert_allclose(list(ours["dcase2020_source"].values())[:4], old.compute_seld_scores(), atol=0, rtol=0)
        np.testing.assert_allclose(list(ours["dcase2023_micro"].values()), modern.compute_seld_scores()[:5], atol=0, rtol=0)

    def test_file_order_does_not_change_counts(self):
        a, b = AlignedMetrics(classes=3), AlignedMetrics(classes=3)
        clips = [({0: [[0, 0, 10., 0.]]}, {0: [[0, 0, 0., 0.]]}),
                 ({599: [[1, 0, 0., 0.]]}, {599: [[1, 0, 0., 0.]]})]
        for pred, gt in clips: a.update(pred, gt)
        for pred, gt in reversed(clips): b.update(pred, gt)
        self.assertEqual(a.scores(), b.scores())

    def test_explicit_csv_formats_and_conversion(self):
        examples = {"polar4": "0,1,90,0\n", "polar5": "0,1,0,90,0\n", "polar6": "0,1,0,90,0,4\n",
                    "cartesian5": "0,1,0,1,0\n", "cartesian6": "0,1,0,0,1,0\n", "cartesian7": "0,1,0,0,1,0,0\n"}
        with tempfile.TemporaryDirectory() as folder:
            for schema, text in examples.items():
                path = Path(folder) / (schema + ".csv")
                path.write_text(text)
                self.assertEqual(load_csv(path, schema), {0: [[1, 0, 90., 0.]]})

    def test_invalid_input_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            for text in ("600,1,90,0\n", "0,14,90,0\n", "0,1,nan,0\n", "0,1,0,90,0\n"):
                path.write_text(text)
                with self.assertRaises(ValueError): load_csv(path, "polar4")
            path.write_text("0,1,0,0,0\n")
            with self.assertRaises(ValueError): load_csv(path, "cartesian5")

    def test_einv2_adapter_returns_true_recall(self):
        pred = {0: [[0, 0, 0., 0.], [1, 0, 0., 0.]]}
        gt = {0: [[0, 0, 0., 0.]]}
        adapter = EINV2Adapter(nb_classes=3)
        adapter.update_seld_scores(segment_labels(pred, keep_track=False), segment_labels(gt, keep_track=False))
        expected = score(pred, gt)["dcase2023_micro"]
        np.testing.assert_allclose(adapter.compute_seld_scores(), [expected[k] for k in ("ER20", "F20", "LE_CD", "LR_CD")], atol=0, rtol=0)
        self.assertAlmostEqual(adapter.compute_seld_scores()[3], 1.)

    def test_einv2_adapter_multisource(self):
        pred = {0: [[0, 0, 10., 0.]]}
        gt = {0: [[0, 2, 0., 0.], [0, 8, 90., 0.]]}
        adapter = EINV2Adapter(nb_classes=3)
        adapter.update_seld_scores(segment_labels(pred, keep_track=False), segment_labels(gt, keep_track=False))
        self.assertAlmostEqual(adapter.compute_seld_scores()[2], 10.)
        self.assertAlmostEqual(adapter.compute_seld_scores()[3], .5)

    def _jackknife_case(self, average):
        path = ROOT.parent / 'models/multi_accdoa/cls_compute_seld_results.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ComputeSELDResults')
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_SELD_Results'))
        jack = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'jackknife_estimation'))
        namespace = dict(np=np, os=os, stats=stats, SELD_evaluation_metrics=OFFICIAL_2023)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[jack, method], type_ignores=[])), str(path), 'exec'), namespace)
        gt = {name: {0: [[0, 0, 0., 0.]]} for name in ('a.csv', 'b.csv', 'c.csv')}
        pred = {'a.csv': gt['a.csv'], 'b.csv': {}, 'c.csv': {0: [[0, 0, 40., 0.], [1, 0, 0., 0.]]}}
        feature = SimpleNamespace(get_nb_classes=lambda: 14,
                                  load_output_format_file=lambda p: pred[Path(p).name],
                                  segment_labels=lambda labels, frames: segment_labels(labels, frames))
        obj = SimpleNamespace(_feat_cls=feature, _ref_labels={name: [segment_labels(labels), 600] for name, labels in gt.items()},
                              _use_polar_format=False, _doa_thresh=20, _average=average)
        with tempfile.TemporaryDirectory() as folder:
            for name in pred: (Path(folder) / name).write_text('')
            plain = namespace['get_SELD_Results'](obj, folder, is_jackknife=False)
            interval = namespace['get_SELD_Results'](obj, folder, is_jackknife=True)
        np.testing.assert_allclose([entry[0] for entry in interval[:5]], plain[:5], atol=0, rtol=0)
        if average == 'macro':
            np.testing.assert_allclose(interval[5][0], plain[5], atol=0, rtol=0)
            self.assertEqual(interval[5][1].shape, (5, 14, 2))

    def test_jackknife_preserves_full_dataset_point(self):
        self._jackknife_case('micro')

    def test_jackknife_macro_supports_14_classes(self):
        self._jackknife_case('macro')


if __name__ == "__main__":
    unittest.main(verbosity=2)
