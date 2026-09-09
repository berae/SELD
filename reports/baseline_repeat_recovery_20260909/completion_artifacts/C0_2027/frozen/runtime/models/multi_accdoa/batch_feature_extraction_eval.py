"""Extract and normalize TAU2020 evaluation features using the dev scaler."""

import sys

import cls_feature_class
import parameters


def main(argv):
    task_id = "32" if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)
    if params["mode"] != "eval":
        raise ValueError("Evaluation feature extraction requires an eval task id")

    eval_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)
    eval_feat_cls.extract_all_feature()
    eval_feat_cls.preprocess_features()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
