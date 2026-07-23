from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common_executable_evaluator import (  # noqa: E402
    DEFAULT_BUDGETS,
    evaluate_common_models,
    load_action_value_predictions,
    load_classification_predictions,
    load_support_audits,
    save_common_evaluation,
    summarize_support_censoring,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BROAD, EXEC_CLS and EXEC_AV on exact sample keys with daily fixed budgets "
            "and a one-position non-overlap constraint."
        )
    )
    parser.add_argument("--broad", type=Path, required=True, help="BROAD validation/test probabilities CSV.")
    parser.add_argument("--exec-cls", type=Path, required=True, help="EXEC_CLS probabilities CSV.")
    parser.add_argument("--exec-av", type=Path, required=True, help="EXEC_AV action-values NPZ.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--broad-up-column", default="p_up")
    parser.add_argument("--broad-down-column", default="p_down")
    parser.add_argument("--exec-up-column", default="p_up")
    parser.add_argument("--exec-down-column", default="p_down")
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--support-audit",
        type=Path,
        nargs="*",
        default=[],
        help="Optional support-audit NPZ files or sequence directories for day/time censor diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic tie-break/bootstrap seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    broad = load_classification_predictions(
        args.broad,
        model="BROAD",
        up_column=args.broad_up_column,
        down_column=args.broad_down_column,
        require_economic=False,
    )
    exec_cls = load_classification_predictions(
        args.exec_cls,
        model="EXEC_CLS",
        up_column=args.exec_up_column,
        down_column=args.exec_down_column,
    )
    exec_av = load_action_value_predictions(args.exec_av)
    result = evaluate_common_models(
        broad,
        exec_cls,
        exec_av,
        budgets=args.budgets,
        seed=args.seed,
    )
    artifacts = save_common_evaluation(result, args.output_dir)
    if args.support_audit:
        audit = load_support_audits(args.support_audit)
        censor_summary = summarize_support_censoring(audit)
        censor_path = args.output_dir / "preprocessing_censor_audit_by_day_time.csv"
        censor_summary.to_csv(censor_path, index=False)
        artifacts["preprocessing_censor_audit"] = censor_path
    print(f"Common three-objective evaluation written to {args.output_dir.resolve()}.")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
