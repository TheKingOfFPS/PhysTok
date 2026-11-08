#!/usr/bin/env python3
"""Verify released aggregate results without raw data, model training, or a GPU."""
import argparse
import json
import os
from pathlib import Path
import sys

# A small analysis should not occupy all CPU threads by default.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phystok.statistics import verify_release


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--mode",choices=["verify","bootstrap","bootstrap-all"],default="verify",
                        help="verify arithmetic, regenerate main 100k intervals, or regenerate main and all control intervals")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"results_verification.json")
    args=parser.parse_args()
    result=verify_release(args.results,bootstrap=args.mode!="verify",bootstrap_ablations=args.mode=="bootstrap-all")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ["status","checks","passed","mode","bootstrap_main_recomputed","bootstrap_ablations_recomputed","source_observations_or_predictions_verified"]}))
    if result["status"]!="PASS":
        for row in result["check_details"]:
            if not row["passed"]:print(json.dumps(row))
        raise SystemExit(1)

if __name__=="__main__":main()
