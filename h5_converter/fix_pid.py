#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_pid.py — .hea / .json 내부의 record 이름·PID 를 파일명 기준으로 정정한다.

원본 .hea 는 첫 줄의 record 이름과 신호 파일 참조('XXX.SIG')가 실제 파일명과 다른
경우가 있다. wfdb 는 .hea 에 적힌 이름으로 신호 파일을 열기 때문에 이 상태로 변환하면
FileNotFoundError 로 실패한다.

  .hea  : 1행 record 이름, 2행~ 'XXX.SIG' 참조를 파일명으로 교체
  .json : Holter Report/PatientInfo/PID 를 파일명 끝 토큰으로 교체

원본을 제자리에서 수정하므로 반드시 복사본에 돌린다. NAS 원본 경로
(/home/coder/workspace/Holter_TOF)는 --allow-original 없이는 거부한다.

예전 버전 대비:
  - 경로를 인자로 받는다 (하드코딩 제거)
  - 하위 디렉토리까지 처리한다 (LQT 의 addition_260729 등)
  - 실제로 다른 경우에만 다시 쓴다. 공백·들여쓰기가 괜히 바뀌지 않는다.
  - .tmp 로 쓴 뒤 원자적으로 교체한다
  - 예외를 조용히 삼키지 않고 개수와 목록을 보고한다
  - --dry-run 으로 몇 개가 바뀔지 먼저 본다

사용:
  python h5_converter/fix_pid.py /home/coder/workspace/data/raw --dry-run
  python h5_converter/fix_pid.py /home/coder/workspace/data/raw
"""

import argparse
import json
import os
import stat
import sys
from collections import Counter

PROTECTED = ("/home/coder/workspace/Holter_TOF",)


def _write_atomic(path, text):
    tmp = path + ".fixpid.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
    except PermissionError:
        raise
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode) | stat.S_IWUSR | stat.S_IRUSR
        os.chmod(tmp, mode)
    except OSError:
        pass
    os.replace(tmp, path)


def fix_hea(hea_path, record_name, dry_run=False):
    """반환: 'changed' | 'ok' (이미 맞음)"""
    with open(hea_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    new_lines, changed = [], False
    for idx, line in enumerate(lines):
        parts = line.split()
        if not parts:
            new_lines.append(line)
            continue
        if idx == 0:
            if parts[0] != record_name:
                parts[0] = record_name
                line = " ".join(parts) + "\n"
                changed = True
        else:
            stem, dot, ext = parts[0].rpartition(".")
            if dot and ext.upper() == "SIG" and parts[0] != f"{record_name}.SIG":
                parts[0] = f"{record_name}.SIG"
                line = " ".join(parts) + "\n"
                changed = True
        new_lines.append(line)
    if changed and not dry_run:
        _write_atomic(hea_path, "".join(new_lines))
    return "changed" if changed else "ok"


def fix_json(json_path, pid, dry_run=False):
    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if "Holter Report" not in data:
        return "no_report"
    info = data["Holter Report"].setdefault("PatientInfo", {})
    if str(info.get("PID", "")) == pid:
        return "ok"
    if not dry_run:
        info["PID"] = pid
        _write_atomic(json_path, json.dumps(data, indent=2, ensure_ascii=False))
    return "changed"


def fix_hea_and_json_by_filename(input_dir, recursive=True, dry_run=False, verbose=True):
    stats, errors, examples = Counter(), [], []
    walker = os.walk(input_dir) if recursive else [(input_dir, [], os.listdir(input_dir))]
    targets = []
    for dirpath, _, names in walker:
        for nm in names:
            if nm.endswith(".hea") or nm.endswith(".json"):
                targets.append(os.path.join(dirpath, nm))
    total = len(targets)
    for i, path in enumerate(sorted(targets)):
        base = os.path.basename(path)
        record_name, ext = os.path.splitext(base)
        pid = record_name.split("_")[-1]
        try:
            if ext == ".hea":
                r = fix_hea(path, record_name, dry_run)
                stats[f".hea {r}"] += 1
            else:
                r = fix_json(path, pid, dry_run)
                stats[f".json {r}"] += 1
            if r == "changed" and len(examples) < 5:
                examples.append(path)
        except Exception as e:
            errors.append((path, f"{type(e).__name__}: {e}"))
            stats[f"{ext} error"] += 1
        if verbose and ((i + 1) % 500 == 0 or i + 1 == total):
            print(f"\r  {i + 1:,}/{total:,}", end="", flush=True)
    if verbose:
        print()
    return stats, errors, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--dry-run", action="store_true", help="바꾸지 않고 개수만 센다")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--allow-original", action="store_true",
                    help="NAS 원본 경로에서도 실행 허용 (권장하지 않음)")
    args = ap.parse_args()

    d = os.path.abspath(args.input_dir)
    if any(d == p or d.startswith(p + os.sep) for p in PROTECTED) and not args.allow_original:
        sys.exit(f"거부: {d} 는 원본 경로입니다. 복사본에 실행하세요 "
                 f"(정말 원본을 수정하려면 --allow-original).")

    print(f"[fix_pid] {d}  {'(dry-run: 수정하지 않음)' if args.dry_run else ''}")
    stats, errors, examples = fix_hea_and_json_by_filename(
        d, recursive=not args.no_recursive, dry_run=args.dry_run)
    verb = "바뀔" if args.dry_run else "바뀐"
    for k in sorted(stats):
        print(f"  {k:<20s} {stats[k]:>7,}")
    print(f"  → {verb} .hea {stats['.hea changed']:,}개 / .json {stats['.json changed']:,}개")
    for p in examples:
        print(f"    예: {p}")
    if errors:
        print(f"  ** 오류 {len(errors):,}건 **")
        for p, e in errors[:10]:
            print(f"    {p}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
