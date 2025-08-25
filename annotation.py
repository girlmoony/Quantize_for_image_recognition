#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Excel の2列（A: 新しいフォルダ名 / B: 現在のフォルダ名）に基づき、
dataset_root/test, /train, /val 配下で一致するフォルダを検索して
  1) BがありAもある → 一致フォルダを新しい名前(A)へリネーム
  2) BがありAが空     → 一致フォルダを削除（中身ごと）
を行います。

安全装置:
- --dry-run で実際には変更せず計画のみ表示
- 競合(同名フォルダの既存/重複一致)や無効値は自動スキップ
- 実行ログCSVを残す (--log)

python folder_ops.py \
  --excel /path/to/mapping.xlsx \
  --sheet Sheet1 \
  --col-a A \
  --col-b B \
  --dataset-root /path/to/dataset_root \
  --dry-run
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from datetime import datetime

SPLITS = ["test", "train", "val"]

def read_mapping(excel_path: Path, sheet: str, col_a: str, col_b: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet, dtype={col_a: str, col_b: str})
    # 余計な空白の除去
    for c in [col_a, col_b]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
        else:
            raise ValueError(f"列 '{c}' がシート '{sheet}' に見つかりません。")
    return df

# def is_blank(x: pd.Series) -> bool:
#     return pd.isna(x) or (str(x).strip() == "") or (x == pd.NA)
def is_blank(x) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in ["nan", "<na>"]

def find_dirs_by_name(root: Path, name: str, max_depth: int = 5) -> List[Path]:
    """
    root 直下から max_depth 階層までで name と一致するディレクトリを探す。
    """
    matches = []
    # 幅優先に近い単純 walk。depth 計算は root.parts との差分で概算
    for dirpath, dirnames, _ in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if len(depth) > max_depth:
            # 深すぎる場合は、この配下の探索を軽く抑制
            # （os.walk はスキップがやや面倒なので、軽めの制御）
            continue
        for d in dirnames:
            if d == name:
                matches.append(Path(dirpath) / d)
    return matches

def safe_rename(src: Path, new_name: str) -> Tuple[bool, str]:
    dst = src.parent / new_name
    if dst.exists():
        return False, f"リネーム先が既に存在: {dst}"
    try:
        src.rename(dst)
        return True, f"RENAMED: {src} -> {dst}"
    except Exception as e:
        return False, f"リネーム失敗: {src} -> {dst} ({e})"

def safe_rmtree(target: Path) -> Tuple[bool, str]:
    try:
        shutil.rmtree(target)
        return True, f"DELETED: {target}"
    except Exception as e:
        return False, f"削除失敗: {target} ({e})"

def process(dataset_root: Path,
            df: pd.DataFrame,
            col_a: str,
            col_b: str,
            dry_run: bool,
            max_depth: int) -> List[dict]:
    logs = []
    for idx, row in df.iterrows():
        a_val = row[col_a]
        b_val = row[col_b]

        # 判定
        a_blank = is_blank(a_val)
        b_blank = is_blank(b_val)

        # Bが空なら何もしない（ユーザー仕様では対象外）
        if b_blank:
            logs.append({
                "row": idx + 2,  # Excel視点でヘッダーが1行と仮定
                "action": "SKIP",
                "reason": "B(現在名)が空",
                "old_name(B)": b_val,
                "new_name(A)": a_val,
                "path": "",
                "result": "N/A"
            })
            continue

        # 検索対象 split を順に処理
        for split in SPLITS:
            split_root = dataset_root / split
            if not split_root.exists():
                logs.append({
                    "row": idx + 2,
                    "action": "SKIP",
                    "reason": f"{split_root} が存在しない",
                    "old_name(B)": b_val,
                    "new_name(A)": a_val,
                    "path": "",
                    "result": "N/A"
                })
                continue

            matches = find_dirs_by_name(split_root, b_val, max_depth=max_depth)

            if len(matches) == 0:
                logs.append({
                    "row": idx + 2,
                    "action": "INFO",
                    "reason": "一致フォルダなし",
                    "old_name(B)": b_val,
                    "new_name(A)": a_val,
                    "path": str(split_root),
                    "result": "N/A"
                })
                continue

            if len(matches) > 1:
                # 複数一致は安全のためスキップ
                for m in matches:
                    logs.append({
                        "row": idx + 2,
                        "action": "SKIP",
                        "reason": "同名フォルダが複数一致(安全のため未処理)",
                        "old_name(B)": b_val,
                        "new_name(A)": a_val,
                        "path": str(m),
                        "result": "N/A"
                    })
                continue

            target = matches[0]

            if a_blank:
                # Aが空 → 削除
                if dry_run:
                    logs.append({
                        "row": idx + 2,
                        "action": "DELETE(DRY-RUN)",
                        "reason": "Aが空のため削除予定",
                        "old_name(B)": b_val,
                        "new_name(A)": "",
                        "path": str(target),
                        "result": "OK(予定)"
                    })
                else:
                    ok, msg = safe_rmtree(target)
                    logs.append({
                        "row": idx + 2,
                        "action": "DELETE",
                        "reason": "Aが空のため削除",
                        "old_name(B)": b_val,
                        "new_name(A)": "",
                        "path": str(target),
                        "result": msg if ok else f"ERROR: {msg}"
                    })
            else:
                # Aあり → リネーム
                if a_val == b_val:
                    logs.append({
                        "row": idx + 2,
                        "action": "SKIP",
                        "reason": "AとBが同名",
                        "old_name(B)": b_val,
                        "new_name(A)": a_val,
                        "path": str(target),
                        "result": "N/A"
                    })
                    continue

                if dry_run:
                    dst = target.parent / a_val
                    exists = dst.exists()
                    logs.append({
                        "row": idx + 2,
                        "action": "RENAME(DRY-RUN)",
                        "reason": "Aが指定されているためリネーム予定",
                        "old_name(B)": b_val,
                        "new_name(A)": a_val,
                        "path": f"{target} -> {dst}",
                        "result": "NG(衝突)" if exists else "OK(予定)"
                    })
                else:
                    ok, msg = safe_rename(target, a_val)
                    logs.append({
                        "row": idx + 2,
                        "action": "RENAME",
                        "reason": "Aが指定されているためリネーム",
                        "old_name(B)": b_val,
                        "new_name(A)": a_val,
                        "path": str(target),
                        "result": msg if ok else f"ERROR: {msg}"
                    })

    return logs

def save_logs(logs: List[dict], out_csv: Path):
    df = pd.DataFrame(logs)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

def main():
    parser = argparse.ArgumentParser(description="Excel列に基づきデータセット内フォルダのリネーム/削除を実施")
    parser.add_argument("--excel", required=True, help="Excelファイルのパス")
    parser.add_argument("--sheet", default=0, help="シート名または番号 (default: 0)")
    parser.add_argument("--col-a", default="A", help="A列（新しいフォルダ名）の列名または見出し (default: 'A')")
    parser.add_argument("--col-b", default="B", help="B列（現在のフォルダ名）の列名または見出し (default: 'B')")
    parser.add_argument("--dataset-root", required=True, help="データセットのルート（test/train/val があるディレクトリ）")
    parser.add_argument("--max-depth", type=int, default=5, help="test/train/val 下を探索する最大階層 (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="変更せず計画のみ表示")
    parser.add_argument("--log", default=None, help="実行ログCSVの出力先 (default: ./folder_ops_log_YYYYmmdd_HHMMSS.csv)")
    args = parser.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not excel_path.exists():
        print(f"[ERROR] Excel が見つかりません: {excel_path}", file=sys.stderr)
        sys.exit(1)
    if not dataset_root.exists():
        print(f"[ERROR] dataset_root が見つかりません: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    try:
        df = read_mapping(excel_path, args.sheet, args.col_a, args.col_b)
    except Exception as e:
        print(f"[ERROR] Excel読込に失敗: {e}", file=sys.stderr)
        sys.exit(1)

    logs = process(dataset_root, df, args.col_a, args.col_b, args.dry_run, args.max_depth)

    # ログ出力
    out_csv = Path(args.log) if args.log else Path(f"./folder_ops_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    try:
        save_logs(logs, out_csv)
        print(f"[INFO] ログ出力: {out_csv}")
    except Exception as e:
        print(f"[WARN] ログ出力に失敗: {e}", file=sys.stderr)

    # 概要
    df_log = pd.DataFrame(logs)
    summary = df_log["action"].value_counts().to_dict() if not df_log.empty else {}
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\n完了。")

if __name__ == "__main__":
    main()
