#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
survey_workspace.py — 작업 디렉토리 전체 구조 파악.

repack/통합 결정을 내리기 전에 "실제로 무엇이 어디에 있는가"를 먼저 본다.
데이터는 거의 읽지 않는다 (샘플 파일 몇 개의 헤더만).

보고 내용
  (1) 디렉토리 트리    깊이 제한, 디렉토리마다 확장자 히스토그램 + 용량
  (2) 원본 WFDB        .dat/.hea/.json 이 남아 있는지, .hea/.json 내부 구조
  (3) 임상 CSV         컬럼, 행수, 고유값, ID 후보 컬럼, 앞부분 미리보기
  (4) record 이름 대조 원본 ↔ h5 코호트 ↔ CSV 사이의 겹침

사용:
  python survey_workspace.py
  python survey_workspace.py --root /home/coder/workspace --depth 3
  python survey_workspace.py --root /home/coder/workspace/Holter_TOF --max-files 500000
"""

import argparse
import collections
import csv as csvmod
import json
import os
import re
import sys

DEFAULT_ROOT = "/home/coder/workspace/Holter_TOF"
BAR = "=" * 78
SIG_EXT = (".h5", ".hdf5", ".dat", ".npy", ".mat")
# 이 데이터셋의 원본은 WFDB 표준 .dat 가 아니라 MARS export 의 .SIG 이고,
# beat 주석은 .ANN 이다 (utils.py 의 has_all_required_files 참고).
RAW_EXT = (".dat", ".sig", ".hea", ".json", ".ann", ".atr", ".ecg", ".xml")


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f}{u}" if u != "B" else f"{int(n)}B"
        n /= 1024.0


def stem(name):
    base = os.path.basename(name)
    root, ext = os.path.splitext(base)
    return root if ext else base


# =============================================================================
# (1) 트리
# =============================================================================
def walk(root, max_depth, max_files):
    """디렉토리별 집계. 깊이를 넘어서면 그 아래는 부모에 합산만 한다."""
    info = {}
    total_files = [0]
    stop = [False]

    def rec(path, depth):
        ext = collections.Counter()
        size = collections.Counter()
        nsub = 0
        try:
            entries = list(os.scandir(path))
        except (PermissionError, OSError) as e:
            info[path] = {"error": str(e), "depth": depth}
            return ext, size, 0
        for e in entries:
            if stop[0]:
                break
            try:
                if e.is_dir(follow_symlinks=False):
                    nsub += 1
                    sub_ext, sub_size, _ = rec(e.path, depth + 1)
                    ext.update(sub_ext)
                    size.update(sub_size)
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            x = os.path.splitext(e.name)[1].lower() or "<noext>"
            ext[x] += 1
            size[x] += st.st_size
            total_files[0] += 1
            if total_files[0] >= max_files:
                stop[0] = True
        if depth <= max_depth:
            info[path] = {"depth": depth, "ext": ext, "size": size, "nsub": nsub}
        return ext, size, nsub

    rec(root, 0)
    return info, total_files[0], stop[0]


def print_tree(info, root):
    print(BAR); print(f"[트리] {root}"); print(BAR)
    for path in sorted(info):
        d = info[path]
        if "error" in d:
            print(f"{'  '*d['depth']}{os.path.basename(path) or path}/  ** {d['error']} **")
            continue
        ext, size = d["ext"], d["size"]
        n, tot = sum(ext.values()), sum(size.values())
        label = os.path.basename(path) or path
        pad = "  " * d["depth"]
        head = f"{pad}{label}/"
        print(f"{head:<46s} 파일 {n:>8,}  {human(tot):>9s}  하위 {d['nsub']}")
        if n:
            top = "  ".join(f"{k} {v:,}" for k, v in ext.most_common(6))
            print(f"{pad}    └ {top}")
    print()


# =============================================================================
# (2) 원본 WFDB
# =============================================================================
def find_raw(root, max_scan=2_000_000):
    """확장자별 파일 목록(상한 있음). 원본이 살아있는지 확인용."""
    found = collections.defaultdict(list)
    n = 0
    for dirpath, _, names in os.walk(root):
        for nm in names:
            x = os.path.splitext(nm)[1].lower()
            if x in RAW_EXT:
                if len(found[x]) < 20000:
                    found[x].append(os.path.join(dirpath, nm))
            n += 1
            if n > max_scan:
                return found, True
    return found, False


def show_hea(path):
    print(f"  --- {os.path.basename(path)} ---")
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 8:
                    print("      ...")
                    break
                print(f"      {line.rstrip()}")
    except OSError as e:
        print(f"      [읽기 실패] {e}")


def json_shape(obj, prefix="", depth=0, max_depth=3, out=None):
    """JSON 구조를 경로 형태로 요약. 값이 아니라 키 구조를 본다."""
    out = [] if out is None else out
    if depth > max_depth:
        return out
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:12]:
            p = f"{prefix}/{k}"
            if isinstance(v, (dict, list)):
                out.append((p, type(v).__name__, len(v)))
                json_shape(v, p, depth + 1, max_depth, out)
            else:
                out.append((p, type(v).__name__, str(v)[:40]))
    elif isinstance(obj, list) and obj:
        out.append((prefix + "[0]", type(obj[0]).__name__, len(obj)))
        json_shape(obj[0], prefix + "[0]", depth + 1, max_depth, out)
    return out


def show_json(path):
    print(f"  --- {os.path.basename(path)} ---")
    try:
        with open(path, "r", errors="replace") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"      [파싱 실패] {type(e).__name__}: {e}")
        return
    for p, t, extra in json_shape(obj)[:40]:
        print(f"      {p:<52s} {t:<6s} {extra}")


# =============================================================================
# (3) 임상 CSV
# =============================================================================
def show_csv(path, keysets=None, preview=4):
    print(f"  --- {os.path.basename(path)} ---")
    try:
        with open(path, "r", errors="replace", newline="") as f:
            sample = f.read(64 * 1024)
            f.seek(0)
            try:
                dialect = csvmod.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except csvmod.Error:
                sep = ","
            rows = list(csvmod.reader(f, delimiter=sep))
    except OSError as e:
        print(f"      [읽기 실패] {e}")
        return None
    if not rows:
        print("      빈 파일")
        return None

    header, body = rows[0], rows[1:]
    print(f"      구분자 '{sep}'   {len(body):,}행 × {len(header)}열")
    print(f"      컬럼: {header}")

    cols = list(zip(*body)) if body else [()] * len(header)
    stats = []
    for i, name in enumerate(header):
        vals = [v.strip() for v in (cols[i] if i < len(cols) else ())]
        nonempty = [v for v in vals if v != ""]
        uniq = len(set(nonempty))
        numeric = sum(1 for v in nonempty[:200] if re.fullmatch(r"-?\d+(\.\d+)?", v))
        kind = "숫자" if nonempty and numeric > len(nonempty[:200]) * 0.9 else "문자"
        stats.append({"col": name, "n": len(nonempty), "uniq": uniq, "kind": kind,
                      "ex": nonempty[:3], "vals": set(nonempty)})

    print(f"      {'컬럼':<22s}{'채움':>8s}{'고유':>8s}{'형':>5s}  예시")
    for s in stats:
        print(f"      {s['col'][:20]:<22s}{s['n']:>8,}{s['uniq']:>8,}{s['kind']:>5s}  "
              f"{', '.join(s['ex'])[:44]}")

    # ID 후보: 거의 전부 고유한 컬럼 (행수가 충분할 때만 의미가 있다)
    cand = [s["col"] for s in stats
            if len(body) >= 5 and s["n"] >= 0.9 * len(body) and s["uniq"] >= 0.9 * s["n"]]
    print(f"      ID 후보 컬럼: {cand if cand else '없음 (행수가 적거나 고유 컬럼 없음)'}")

    if keysets:
        # 레코드가 여러 코호트에 흩어져 있으므로 합집합 기준도 함께 본다.
        # '<...>' 로 시작하는 키는 파생 키(끝토큰 등)이므로 합집합에서 제외한다.
        real = [v for k, v in keysets.items() if not k.startswith("<")]
        ks_all = dict(keysets)
        if real:
            ks_all["<h5 전체 합집합>"] = set().union(*real)
        print("      h5 record 이름과의 매칭 (CSV 값 중 h5 에 존재하는 비율):")
        any_hit = False
        for s in stats:
            if s["uniq"] < 2:
                continue
            hits = [(k, len(s["vals"] & ks)) for k, ks in ks_all.items()]
            hits = [(k, n) for k, n in hits if n > 0]
            if not hits:
                continue
            any_hit = True
            hits.sort(key=lambda t: -t[1])
            detail = "  ".join(f"{k} {n:,}" for k, n in hits[:4])
            top = hits[0][1] / max(len(s["vals"]), 1)
            print(f"        {s['col'][:24]:<26s} 최대 {top:>6.1%}   {detail}")
        if not any_hit:
            print("        ** 어느 컬럼도 h5 record 이름과 겹치지 않음 — "
                  "ID 체계가 다르거나 접두/접미가 붙어 있을 수 있음 **")

    print("      미리보기:")
    for r in body[:preview]:
        print("        " + " | ".join(x[:18] for x in r[:8]))
    return {"header": header, "n": len(body), "stats": stats}


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-files", type=int, default=3_000_000)
    ap.add_argument("--no-raw", action="store_true", help="원본 탐색 생략")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"디렉토리 없음: {args.root}")

    info, nfiles, truncated = walk(args.root, args.depth, args.max_files)
    print_tree(info, args.root)
    if truncated:
        print(f"** --max-files {args.max_files:,} 도달, 일부만 집계됨 **\n")

    # h5 코호트의 record 이름 수집 (CSV/원본 대조용)
    keysets = {}
    for path, d in info.items():
        if "ext" not in d or d["depth"] == 0:
            continue
        if not any(d["ext"].get(x) for x in (".h5", ".hdf5")):
            continue
        try:
            names = {stem(e.name) for e in os.scandir(path)
                     if e.is_file() and e.name.lower().endswith((".h5", ".hdf5"))}
        except OSError:
            continue
        if names:
            keysets[os.path.basename(path)] = names

    # 파일명이 DVD20130523_8_2522134 처럼 '..._PID' 형태라(fix_pid.py 참고)
    # PID 로만 작성된 임상 CSV 는 원본 이름과 직접 매칭되지 않는다.
    # 끝 토큰과 첫 토큰을 파생 키로 만들어 함께 대조한다.
    if keysets:
        allnames = set().union(*keysets.values())
        tail = {n.rsplit("_", 1)[-1] for n in allnames if "_" in n}
        head = {n.split("_", 1)[0] for n in allnames if "_" in n}
        if tail:
            keysets["<파일명 끝토큰(PID추정)>"] = tail
        if head:
            keysets["<파일명 첫토큰>"] = head

    # ---- (2) 원본 ----
    if not args.no_raw:
        print(BAR); print("[원본 파일 (.dat/.hea/.json ...)]"); print(BAR)
        raw, cut = find_raw(args.root)
        if not raw:
            print("  원본 형식 파일을 찾지 못했습니다. h5 만 남아 있는 것으로 보입니다.\n")
        else:
            for x in sorted(raw):
                paths = raw[x]
                dirs = collections.Counter(os.path.dirname(p) for p in paths)
                print(f"  {x:<7s} {len(paths):>8,}개   위치 {len(dirs)}곳")
                for d, n in dirs.most_common(3):
                    print(f"           {d}  ({n:,})")
            if cut:
                print("  ** 스캔 상한 도달 **")
            print()
            for x, fn in ((".hea", show_hea), (".json", show_json)):
                if raw.get(x):
                    print(f"  [{x} 내부 구조]")
                    fn(raw[x][0])
                    print()
            # 원본 ↔ h5 대조
            raw_stems = {stem(p) for x in (".dat", ".hea") for p in raw.get(x, [])}
            if raw_stems and keysets:
                print("  [원본 ↔ h5 코호트 대조]")
                print(f"    원본 record stem {len(raw_stems):,}개")
                for name, ks in keysets.items():
                    inter = len(raw_stems & ks)
                    print(f"    {name:<28s} h5 {len(ks):>6,}개 중 원본 보유 {inter:>6,} "
                          f"({100*inter/max(len(ks),1):.0f}%)")
                print()

    # ---- (3) 임상 CSV ----
    print(BAR); print("[CSV]"); print(BAR)
    csvs = []
    for dirpath, _, names in os.walk(args.root):
        for nm in names:
            if nm.lower().endswith(".csv"):
                csvs.append(os.path.join(dirpath, nm))
        if len(csvs) > 50:
            break
    if not csvs:
        print("  CSV 없음\n")
    for p in sorted(csvs)[:12]:
        show_csv(p, keysets)
        print()


if __name__ == "__main__":
    main()
