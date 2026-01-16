#!/usr/bin/env python3
"""
Collect paper-ready artifacts from HierFL runs.

What it produces (under --out_dir):
  - runs_index.csv                 : run metadata (seed/policy/oracle mapping)
  - eval_table.csv                 : parsed "Control-plane Closed-loop Summary" + regret block (all keys)
  - args_table.csv                 : flattened run_args.json for reproducibility
  - train_summary_table.csv        : summary.json fields (training-side summary)
  - cp_curve_long.csv              : cp_metrics.csv as long table (for plotting)
  - train_curve_long.csv           : training_metrics.csv as long table (for plotting)
  - cp_curve_agg.csv               : extra aggregates computed from cp_metrics.csv (risk/trigger/delivered rounds)
  - paper_results_book.xlsx        : OPTIONAL workbook with multiple sheets (use --make_xlsx)

Usage example:
  python collect_paper_artifacts.py --runs "runs/PAPER_*" --ttr_threshold 0.20 --phase2 0.70 --out_dir artifacts --make_xlsx --save_eval_txt

Notes:
  - CSV cannot contain multiple tables. This script therefore writes multiple CSVs, plus (optionally) a single XLSX with separate sheets.
  - It can either RUN eval_controlplane_metrics.py (default) or reuse existing eval_summary.txt (use --reuse_eval_txt).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

SUMMARY_HEADER = "=== Control-plane Closed-loop Summary ==="
REGRET_HEADER = "--- regret vs oracle ---"

def _safe_load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
        return {"_non_dict_json": obj}
    except Exception as e:
        return {"_json_load_error": f"{type(e).__name__}: {e}"}

def _parse_seed(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None

def _infer_policy(name: str) -> str:
    n = name.lower()
    if "oracle" in n: return "ORACLE"
    if "random" in n: return "RANDOM"
    if "ace" in n: return "ACE"
    if "dcas" in n: return "DCAS"
    if "obcmpc" in n: return "OBC-MPC"
    return "OTHER"

def _cast_value(v: str) -> Any:
    v = v.strip()
    # ints
    if re.fullmatch(r"[-+]?\d+", v):
        try: return int(v)
        except: pass
    # floats
    try:
        return float(v)
    except:
        return v

def _run_eval(run_dir: str, oracle_dir: Optional[str], ttr_threshold: float, phase2: float) -> Tuple[int, str]:
    cmd = [sys.executable, "eval_controlplane_metrics.py",
           "--run_dir", run_dir,
           "--ttr_threshold", str(ttr_threshold),
           "--phase2", str(phase2)]
    if oracle_dir and os.path.abspath(oracle_dir) != os.path.abspath(run_dir):
        cmd += ["--oracle_dir", oracle_dir]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def _parse_eval_text(txt: str) -> Dict[str, Any]:
    """
    Parses both blocks:
      1) Closed-loop summary
      2) Regret vs oracle (if present)
    """
    out: Dict[str, Any] = {}
    if SUMMARY_HEADER not in txt:
        out["_eval_parse_error"] = "missing_summary_header"
        out["_eval_output_tail"] = txt[-2000:]
        return out

    in_summary = False
    in_regret = False

    for line in txt.splitlines():
        s = line.strip()
        if s == SUMMARY_HEADER:
            in_summary = True
            in_regret = False
            continue
        if s == REGRET_HEADER:
            in_regret = True
            in_summary = False
            continue

        if ":" not in line:
            continue

        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()

        if in_summary:
            out[k] = _cast_value(v)
        elif in_regret:
            out[f"regret_{k}"] = _cast_value(v)

    return out

def _read_csv_long(path: str, run_name: str, prefix: str) -> List[Dict[str, Any]]:
    """
    Reads a CSV with a 'round' column and returns long rows with run_name.
    """
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out = {"name": run_name}
            for k, v in row.items():
                out[f"{prefix}{k}"] = v
            rows.append(out)
    return rows

def _aggregate_cp_metrics(cp_long_rows: List[Dict[str, Any]], prefix: str="cp_") -> Dict[str, Any]:
    """
    Extra aggregates computed from cp_metrics.csv.
    Expects long rows created by _read_csv_long(..., prefix='cp_') where keys include:
      cp_bytes_delivered, cp_bytes_sent, cp_triggered_edges, cp_risk_sum_topB_mean, cp_risk_max_mean
    """
    def _to_float(x: Any) -> float:
        try: return float(x)
        except: return 0.0
    def _to_int(x: Any) -> int:
        try: return int(float(x))
        except: return 0

    if not cp_long_rows:
        return {}

    delivered_rounds = 0
    sent_rounds = 0
    trig_edges_total = 0
    risk_sum_vals: List[float] = []
    risk_max_vals: List[float] = []

    for r in cp_long_rows:
        bd = _to_int(r.get(f"{prefix}bytes_delivered", 0))
        bs = _to_int(r.get(f"{prefix}bytes_sent", 0))
        te = _to_int(r.get(f"{prefix}triggered_edges", 0))
        delivered_rounds += 1 if bd > 0 else 0
        sent_rounds += 1 if bs > 0 else 0
        trig_edges_total += te
        risk_sum_vals.append(_to_float(r.get(f"{prefix}risk_sum_topB_mean", 0.0)))
        risk_max_vals.append(_to_float(r.get(f"{prefix}risk_max_mean", 0.0)))

    n = len(cp_long_rows)
    return {
        "cp_delivered_rounds": delivered_rounds,
        "cp_sent_rounds": sent_rounds,
        "cp_triggered_edges_total": trig_edges_total,
        "cp_risk_sum_topB_mean_avg": sum(risk_sum_vals)/n if n else 0.0,
        "cp_risk_max_mean_avg": sum(risk_max_vals)/n if n else 0.0,
    }

def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        # write empty with no headers
        with open(path, "w", newline="") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def _write_xlsx(out_path: str,
                sheets: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        from openpyxl import Workbook
    except Exception:
        print("openpyxl not available; skipping xlsx.")
        return

    wb = Workbook()
    # remove default sheet
    wb.remove(wb.active)

    for sheet_name, rows in sheets.items():
        ws = wb.create_sheet(title=sheet_name[:31])
        if not rows:
            ws.append(["(empty)"])
            continue
        keys = sorted({k for r in rows for k in r.keys()})
        ws.append(keys)
        for r in rows:
            ws.append([r.get(k, "") for k in keys])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help='Glob patterns for run directories, e.g. "runs/PAPER_*"')
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--ttr_threshold", type=float, default=0.20)
    ap.add_argument("--phase2", type=float, default=0.70)
    ap.add_argument("--reuse_eval_txt", action="store_true",
                    help="If eval_summary.txt exists in run_dir, parse it instead of running eval script.")
    ap.add_argument("--save_eval_txt", action="store_true",
                    help="Save eval output to eval_summary.txt in each run_dir.")
    ap.add_argument("--make_xlsx", action="store_true",
                    help="Also write paper_results_book.xlsx with multiple sheets.")
    args = ap.parse_args()

    # Discover run dirs
    run_dirs: List[str] = []
    for pat in args.runs:
        run_dirs.extend([d for d in glob.glob(pat) if os.path.isdir(d)])
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        raise SystemExit("No run dirs matched.")

    # Oracle map seed -> oracle dir
    oracle_map: Dict[int, str] = {}
    for d in run_dirs:
        name = os.path.basename(d)
        if "oracle" in name.lower():
            s = _parse_seed(name)
            if s is not None:
                oracle_map[s] = d

    index_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    args_rows: List[Dict[str, Any]] = []
    train_summary_rows: List[Dict[str, Any]] = []
    cp_long: List[Dict[str, Any]] = []
    train_long: List[Dict[str, Any]] = []
    cp_agg_rows: List[Dict[str, Any]] = []

    for d in run_dirs:
        name = os.path.basename(d)
        seed = _parse_seed(name)
        policy = _infer_policy(name)
        oracle_dir = oracle_map.get(seed, "")

        # index
        index_rows.append({
            "name": name,
            "run_dir": d,
            "seed": seed,
            "policy": policy,
            "oracle_dir": oracle_dir,
        })

        # run args
        run_args = _safe_load_json(os.path.join(d, "run_args.json"))
        args_row = {"name": name}
        for k, v in run_args.items():
            args_row[k] = v
        args_rows.append(args_row)

        # training summary.json
        summary = _safe_load_json(os.path.join(d, "summary.json"))
        ts_row = {"name": name}
        for k, v in summary.items():
            ts_row[k] = v
        train_summary_rows.append(ts_row)

        # curves
        cp_path = os.path.join(d, "cp_metrics.csv")
        tr_path = os.path.join(d, "training_metrics.csv")
        cp_rows = _read_csv_long(cp_path, name, prefix="cp_")
        tr_rows = _read_csv_long(tr_path, name, prefix="tr_")
        cp_long.extend(cp_rows)
        train_long.extend(tr_rows)

        # cp aggregates
        agg = _aggregate_cp_metrics(cp_rows, prefix="cp_")
        agg_row = {"name": name, **agg}
        cp_agg_rows.append(agg_row)

        # eval
        eval_txt_path = os.path.join(d, "eval_summary.txt")
        txt = None
        rc = 0
        if args.reuse_eval_txt and os.path.exists(eval_txt_path):
            with open(eval_txt_path, "r") as f:
                txt = f.read()
        else:
            rc, txt = _run_eval(d, oracle_dir if oracle_dir else None, args.ttr_threshold, args.phase2)
            if args.save_eval_txt:
                try:
                    with open(eval_txt_path, "w") as f:
                        f.write(txt)
                except Exception:
                    pass

        eval_data = _parse_eval_text(txt or "")
        eval_row = {"name": name, "eval_returncode": rc}
        # keep oracle_dir for convenience in the eval table too
        eval_row["oracle_dir"] = oracle_dir
        for k, v in eval_data.items():
            eval_row[k] = v
        eval_rows.append(eval_row)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    _write_csv(os.path.join(out_dir, "runs_index.csv"), index_rows)
    _write_csv(os.path.join(out_dir, "eval_table.csv"), eval_rows)
    _write_csv(os.path.join(out_dir, "args_table.csv"), args_rows)
    _write_csv(os.path.join(out_dir, "train_summary_table.csv"), train_summary_rows)
    _write_csv(os.path.join(out_dir, "cp_curve_long.csv"), cp_long)
    _write_csv(os.path.join(out_dir, "train_curve_long.csv"), train_long)
    _write_csv(os.path.join(out_dir, "cp_curve_agg.csv"), cp_agg_rows)

    if args.make_xlsx:
        xlsx_path = os.path.join(out_dir, "paper_results_book.xlsx")
        _write_xlsx(xlsx_path, {
            "index": index_rows,
            "eval": eval_rows,
            "args": args_rows,
            "train_summary": train_summary_rows,
            "cp_agg": cp_agg_rows,
            "cp_curve": cp_long,
            "train_curve": train_long,
        })
        print(f"Wrote XLSX: {xlsx_path}")

    print(f"Done. Outputs written under: {out_dir}")
    print("Download example (Mac):")
    print(f"  scp cai9910@134.28.77.22:~/projectsAnton/Last/{out_dir}/paper_results_book.xlsx .")
    print(f"  scp -r cai9910@134.28.77.22:~/projectsAnton/Last/{out_dir} .")

if __name__ == "__main__":
    main()
