import json
import os
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    root_dir = "/home/popis/Documents/masked-face-recognizer/output/original/1635408880"
    summary = {}
    for directory in os.listdir(root_dir):
        results_fp = os.path.join(root_dir, directory, "results.csv")
        if os.path.exists(results_fp):
            # results = json.load(results_fp)
            df = pd.read_csv(results_fp)
            top5_n = df["top5"].sum()
            perf_n = df["perfect"].sum()
            fail_n = df["fail"].sum()
            tests_number = float(top5_n + perf_n + fail_n)
            subsummary = {
                "tests_number": tests_number,
                "top5": float((top5_n + perf_n) / tests_number * 100),
                "perfect": float(perf_n / tests_number * 100),
            }

            summary[directory] = subsummary
            with open(Path(results_fp).parent / "summary.json", "w+") as fw:
                json.dump(subsummary, fw)

    with open(os.path.join(root_dir, "summary.json"), "w+") as fw:
        json.dump(summary, fw)
