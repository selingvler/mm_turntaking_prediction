import os
import sys
import torch
import copy
import argparse
from sklearn import metrics

if not torch.cuda.is_available():
    torch.cuda.set_device = lambda device: None
    torch.cuda.is_available = lambda: False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from turn_taking.analysis.validation.validation import (
    get_preds,
    optimal_thresholds,
    apply_thresholds,
    apply_score,
    f1_score_func,
    f1_score_func_detailed
)

# MM-VAP task semantic mappings
TASK_LABELS = {
    "shift_hold_p_future": ("hold", "shift"),
    "shift_hold_p_now": ("hold", "shift"),

    "s_pred_p_future": ("negative", "shift"),
    "s_pred_p_now": ("negative", "shift"),

    "backchannel": ("non-bc", "backchannel"),

    "short_long": ("long", "short"),

    "overlaps_before_p_now": ("overlap_hold", "overlap_shift"),
    "overlaps_before_p_future": ("overlap_hold", "overlap_shift"),

    "overlaps_after_p_now": ("overlap_hold", "overlap_shift"),
    "overlaps_after_p_future": ("overlap_hold", "overlap_shift"),

    "overlap_spred_before_p_future": ("negative", "overlap_shift"),

    "gap_0_p_future": ("hold", "shift"),
    "gap_0_p_now": ("hold", "shift"),

    "superset_p_future": ("hold_like", "shift_like"),
}


def evaluate_final_detailed(file_id):

    output_dir = "my_outputs/"

    print(f"\nEvaluating file: {file_id}")

    try:

        y_true, y_score = get_preds(
            [file_id],
            output_dir,
            output_dir
        )

        if not y_true:
            print("No matching prediction/json pair found.")
            return

        print("\nOptimizing thresholds for weighted F1...")

        # ---------------------------------------------------
        # F1 OPTIMIZATION
        # ---------------------------------------------------
        y_score_f1 = copy.deepcopy(y_score)

        thresholds_f1 = optimal_thresholds(
            y_true,
            y_score_f1,
            func_to_optimise=f1_score_func
        )

        y_pred_f1 = apply_thresholds(
            y_score_f1,
            thresholds_f1
        )

        f1_results = apply_score(
            y_true,
            y_pred_f1,
            f1_score_func_detailed
        )

        # ---------------------------------------------------
        # BALANCED ACCURACY
        # ---------------------------------------------------
        y_score_bal = copy.deepcopy(y_score)

        thresholds_bal = optimal_thresholds(
            y_true,
            y_score_bal,
            func_to_optimise=metrics.balanced_accuracy_score
        )

        y_pred_bal = apply_thresholds(
            y_score_bal,
            thresholds_bal
        )

        bal_acc_results = apply_score(
            y_true,
            y_pred_bal,
            metrics.balanced_accuracy_score
        )

        print("\n" + "=" * 70)
        print(f"MM-VAP VALIDATION RESULTS :: {file_id}")
        print("=" * 70)

        for event_class in y_true.keys():

            print(f"\n[{event_class.upper()}]")

            for sub_event in y_true[event_class].keys():

                if len(y_true[event_class][sub_event]) == 0:
                    continue

                weighted_f1, class0_f1, class1_f1 = \
                    f1_results[event_class][sub_event][0]

                bal_acc = \
                    bal_acc_results[event_class][sub_event][0]

                threshold_used = \
                    thresholds_f1[event_class][sub_event]

                n_samples = len(y_true[event_class][sub_event])

                label0, label1 = TASK_LABELS.get(
                    sub_event,
                    ("class0", "class1")
                )

                print(f"\n  Task: {sub_event}")
                print(f"    Samples        : {n_samples}")
                print(f"    Threshold      : {threshold_used:.4f}")

                print(f"    Weighted F1    : {weighted_f1:.4f}")
                print(f"    Balanced Acc   : {bal_acc:.4f}")

                print(f"    F1 [{label0}]  : {class0_f1:.4f}")
                print(f"    F1 [{label1}]  : {class1_f1:.4f}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate MM-VAP predictions."
    )

    parser.add_argument(
        "file_id",
        type=str,
        help="File ID without extension"
    )

    args = parser.parse_args()

    evaluate_final_detailed(args.file_id)