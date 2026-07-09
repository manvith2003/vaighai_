"""DATA QUALITY — hard gates fail the pipeline, soft checks become warnings.
Writes docs/validation_report.md.
"""
import os

import pandas as pd

from utils import ROOT, SILVER, QTY_COLS, banner


def run():
    banner("DQ CHECKS", "Quality gates on the Silver layer")
    panel = pd.read_csv(os.path.join(SILVER, "supply_panel.csv"))
    hard, soft = [], []

    def check(ok, msg, level="HARD"):
        (hard if level == "HARD" else soft).append((ok, msg))
        print(f"  [{'PASS' if ok else level}] {msg}")

    check(len(panel) > 1000, f"panel has {len(panel):,} rows (>1,000 expected)")
    check(panel["supplier"].notna().all(), "no null supplier names")
    dup = panel.duplicated(["supplier", "region", "fiscal_year", "fiscal_quarter"]).sum()
    check(dup == 0, f"no duplicate supplier-quarters (found {dup})")
    for c in QTY_COLS:
        check((panel[c] >= 0).all(), f"{c}: no negative quantities")
    check(panel["fiscal_quarter"].isin(["FQ1", "FQ2", "FQ3", "FQ4"]).all(),
          "fiscal_quarter values all valid")
    wq = pd.read_csv(os.path.join(SILVER, "weather_quarterly.csv"))
    check(len(wq) > 0 and (wq["rain_mm"] >= 0).all(), "weather actuals present, non-negative")

    unk = int((panel["region"] == "UNKNOWN").sum())
    check(unk < len(panel) * 0.05, f"UNKNOWN-region rows: {unk} ({unk/len(panel)*100:.1f}%)", "WARN")
    share_bad = int((panel["our_share_of_dispatch_pct"] > 100).sum())
    check(share_bad == 0, f"share>100% rows after capping: {share_bad}", "WARN")
    one_sided = int((panel["in_mir"] & ~panel["in_purchase"]).sum())
    check(True, f"MIR-only supplier-quarters (estimate vs actual gap): {one_sided:,}", "WARN")

    with open(os.path.join(ROOT, "docs", "validation_report.md"), "w") as f:
        f.write("# Data Validation Report\n\n")
        f.write(f"Generated: {pd.Timestamp.today():%Y-%m-%d %H:%M}\n\n## Hard gates\n\n")
        for ok, msg in hard:
            f.write(f"- {'✅' if ok else '❌'} {msg}\n")
        f.write("\n## Warnings / observations\n\n")
        for ok, msg in soft:
            f.write(f"- {'✅' if ok else '⚠️'} {msg}\n")

    failed = [m for ok, m in hard if not ok]
    if failed:
        raise SystemExit(f"VALIDATION FAILED: {failed}")
    print("  -> docs/validation_report.md; all hard gates passed")


if __name__ == "__main__":
    run()
