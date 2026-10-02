#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_share.py — 원본 + v2 h5 + 라벨 + 코드를 공유용 폴더 하나로 복사한다.

원본은 읽기만 한다. 결과 구조:

  <dest>/
  ├─ README.md            docs/SHARE_README.md 복사본
  ├─ raw/<코호트>/...      --raw 의 record(.hea+.SIG 짝)의 .hea/.SIG/.ANN/.json
  ├─ h5/*.h5              --h5 바로 아래의 .h5 + conversion_log.csv (하위 디렉토리는 제외)
  ├─ labels/              splits / manifest / duplicates / 임상 CSV
  └─ code/holter_psvt/    이 저장소의 HEAD (git archive, log.txt 제외)

  - 대용량 복사는 copy_raw.py 를 그대로 쓴다 → .part 원자적 교체, 재실행하면 이어서 진행
  - labels/ 의 splits.csv · manifest.* 는 path 열을 <dest>/h5/<파일명> 으로 바꿔 쓴다
    (원본 표는 건드리지 않는다)
  - 없는 라벨 파일은 경고만 하고 넘어간다

사용 (서버에서, 저장소 루트에서):
  python tools/make_share.py --dest /home/coder/workspace/Holter_TOF/holter_total --dry-run
  python tools/make_share.py --dest /home/coder/workspace/Holter_TOF/holter_total --workers 8
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BAR = "=" * 78

# h5 디렉토리(--h5)에서 가져올 표
FROM_H5 = ("splits.csv", "splits_summary.txt", "manifest.csv", "manifest.parquet",
           "report_features.csv")
# 임상 CSV 디렉토리(--clinical-dir)에서 가져올 표
FROM_CLINICAL = ("clinical_data_psvt.csv", "clinical_data_tof.csv", "psvt_labeling.csv")
# path 열을 새 위치로 바꿔 쓸 표
REPATH = ("splits.csv", "manifest.csv", "manifest.parquet")


def find_duplicates_csv(explicit, roots):
    """find_duplicates.py 의 기본 출력(duplicates.csv)을 흔한 위치에서 찾는다."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for r in roots:
        p = os.path.join(r, "duplicates.csv")
        if os.path.isfile(p):
            return p
    return None


def copy_labels(args, lab, dry):
    print(f"\n{BAR}\n[라벨] → {lab}\n{BAR}")
    srcs = [os.path.join(args.h5, n) for n in FROM_H5]
    srcs += [os.path.join(args.clinical_dir, n) for n in FROM_CLINICAL]
    dup = find_duplicates_csv(args.duplicates,
                              [args.h5, os.path.dirname(args.h5), REPO, os.getcwd()])
    srcs.append(dup or "duplicates.csv")
    srcs += args.extra_labels

    missing = []
    for s in srcs:
        if not os.path.isfile(s):
            missing.append(s)
            print(f"  [없음] {s}")
            continue
        print(f"  {s}")
        if not dry:
            os.makedirs(lab, exist_ok=True)
            shutil.copy2(s, os.path.join(lab, os.path.basename(s)))
    if missing:
        print(f"  ** {len(missing)}개 없음 — 위치가 다르면 --duplicates / --extra-labels 로 지정 **")


def repath_labels(lab, h5_dest):
    """labels/ 복사본의 path 열을 새 h5 위치로 바꾼다.

    csv 는 path 열만 바꾸고 나머지 칸은 글자 그대로 둔다 (pandas 로 왕복하면
    float 표기·빈 칸·pid 앞자리 0 이 바뀔 수 있다).
    """
    def new_path(v):
        return os.path.join(h5_dest, os.path.basename(v)) if v else v

    for name in REPATH:
        p = os.path.join(lab, name)
        if not os.path.isfile(p):
            continue
        tmp = p + ".tmp"
        if p.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(p)
            if "path" not in df.columns:
                print(f"  {name}: path 열 없음 — 그대로 둠")
                continue
            df["path"] = df["path"].astype(str).map(new_path)
            paths = list(df["path"])
            df.to_parquet(tmp, index=False)
        else:
            with open(p, newline="", encoding="utf-8") as fi:
                rows = list(csv.reader(fi))
            if not rows or "path" not in rows[0]:
                print(f"  {name}: path 열 없음 — 그대로 둠")
                continue
            j = rows[0].index("path")
            for r in rows[1:]:
                if j < len(r):
                    r[j] = new_path(r[j])
            paths = [r[j] for r in rows[1:] if j < len(r)]
            with open(tmp, "w", newline="", encoding="utf-8") as fo:
                csv.writer(fo).writerows(rows)
        os.replace(tmp, p)
        n_ok = sum(os.path.isfile(x) for x in paths)
        print(f"  {name}: path → {h5_dest}/  (파일 존재 {n_ok:,}/{len(paths):,})")


def copy_code(code_dest, dry):
    print(f"\n{BAR}\n[코드] {REPO} HEAD → {code_dest}/holter_psvt\n{BAR}")
    git = ["git", "-C", REPO]
    head = subprocess.run(git + ["log", "-1", "--format=%h %ad %s", "--date=short"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    print(f"  HEAD {head}")
    if dirty:
        print("  ** 커밋 안 된 변경이 있습니다. git archive 는 HEAD 만 담으므로 빠집니다 **")
    if dry:
        return
    dst = os.path.join(code_dest, "holter_psvt")
    if os.path.isdir(dst):
        shutil.rmtree(dst)                         # 이전 스냅샷을 통째로 교체
    os.makedirs(code_dest, exist_ok=True)
    arc = subprocess.Popen(git + ["archive", "--prefix=holter_psvt/", "HEAD"],
                           stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", code_dest], stdin=arc.stdout, check=True)
    if arc.wait() != 0:
        sys.exit("git archive 실패")
    for junk in ("log.txt",):                      # 커밋돼 있지만 공유할 필요 없는 것
        p = os.path.join(dst, junk)
        if os.path.exists(p):
            os.remove(p)


def run_copy(cmd):
    print("  $ " + " ".join(cmd))
    r = subprocess.run([sys.executable, os.path.join(HERE, "copy_raw.py")] + cmd)
    if r.returncode != 0:
        sys.exit(f"copy_raw.py 실패 (exit {r.returncode}). 다시 실행하면 이어서 진행합니다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--raw", default="/home/coder/workspace/data/raw",
                    help="fix_pid 적용된 원본 루트 (아래에 코호트 디렉토리)")
    ap.add_argument("--h5", default="/home/coder/workspace/data/holter_v2",
                    help="v2 h5 디렉토리 (splits.csv, manifest.* 도 여기서 찾는다)")
    ap.add_argument("--clinical-dir", default="/home/coder/workspace/Holter_TOF")
    ap.add_argument("--duplicates", default=None, help="find_duplicates.py 결과 CSV")
    ap.add_argument("--extra-labels", nargs="*", default=[], help="labels/ 에 더 넣을 파일")
    ap.add_argument("--skip", nargs="*", default=[], choices=["raw", "h5", "labels", "code"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="대용량 복사 후 sha256 비교 (느림)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)    # 자식(copy_raw) 출력과 순서가 섞이지 않게

    dest = os.path.abspath(args.dest)
    for src in (args.raw, args.h5):
        if os.path.commonpath([os.path.abspath(src), dest]) == os.path.abspath(src):
            sys.exit(f"--dest 가 원본({src}) 안에 있습니다.")
    h5_dest = os.path.join(dest, "h5")
    lab = os.path.join(dest, "labels")
    dry = args.dry_run
    extra = (["--workers", str(args.workers)] + (["--verify"] if args.verify else [])
             + (["--dry-run"] if dry else []))

    # 작은 것부터: 구조와 라벨을 먼저 만들어 두면 대용량 복사가 길어져도 바로 확인할 수 있다
    if "labels" not in args.skip:
        copy_labels(args, lab, dry)
    if "code" not in args.skip:
        copy_code(os.path.join(dest, "code"), dry)
        readme = os.path.join(REPO, "docs", "SHARE_README.md")
        if not dry and os.path.isfile(readme):
            shutil.copy2(readme, os.path.join(dest, "README.md"))

    if "raw" not in args.skip:
        print(f"\n{BAR}\n[원본] {args.raw} → {dest}/{os.path.basename(args.raw.rstrip('/'))}\n{BAR}")
        run_copy(["--src", args.raw, "--dest", dest, "--records-only"] + extra)

    if "h5" not in args.skip:
        print(f"\n{BAR}\n[h5] {args.h5}/*.h5 → {h5_dest}\n{BAR}")
        names = sorted(e.name for e in os.scandir(args.h5)
                       if e.is_file() and (e.name.endswith(".h5") or e.name == "conversion_log.csv"))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(names) + "\n")
            listing = f.name
        try:
            run_copy(["--files-from", listing, "--base", args.h5, "--dest", h5_dest] + extra)
        finally:
            os.remove(listing)

    # path 는 h5 복사가 끝난 뒤에 바꿔야 존재 여부 집계가 맞다
    if not dry and "labels" not in args.skip:
        print(f"\n{BAR}\n[라벨 path 갱신]\n{BAR}")
        repath_labels(lab, h5_dest)

    print(f"\n완료{' (dry-run)' if dry else ''}: {dest}")


if __name__ == "__main__":
    main()
