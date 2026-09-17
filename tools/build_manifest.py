#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_manifest.py — v2 레코드 전체를 훑어 단일 meta 파일을 만든다.

레코드마다 h5 를 열어 라벨과 메타를 찾아다니는 대신, 여기서 만든 manifest 하나로
split 구성 / 필터링 / 라벨 조인을 모두 해결한다. 신호는 읽지 않으므로 빠르다
(root attrs + 작은 파생 배열만 읽는다).

임상 CSV(clinical_data_psvt.csv, clinical_data_tof.csv)는 ID 컬럼을 자동 탐지해
붙인다. record_name 과 pid 양쪽에 대해 매칭률을 계산하고 가장 잘 맞는 조합을 고른다.

사용:
  python build_manifest.py --src /scratch/holter_v2 \
      --clinical psvt=/home/coder/workspace/Holter_TOF/clinical_data_psvt.csv \
                 tof=/home/coder/workspace/Holter_TOF/clinical_data_tof.csv \
      --out /scratch/holter_v2/manifest
"""

import argparse
import json
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from h5_converter.schema_v2 import WFDB_SYMBOLS, _to_str  # noqa: E402

# 임상적으로 의미가 큰 beat 심볼만 별도 컬럼으로 뺀다.
#   N 정상, S 상심실성(조기), V 심실성, A 심방조기, F 융합, Q 미분류, / 페이싱
KEY_SYMBOLS = ["N", "S", "V", "A", "F", "Q", "/"]


def _j(v, default=None):
    """JSON 문자열로 저장된 attr 을 되돌린다."""
    if isinstance(v, (bytes, str)):
        try:
            return json.loads(_to_str(v))
        except (ValueError, TypeError):
            return _to_str(v)
    return default if v is None else v


def scan_record(path):
    row = {"path": path, "file_bytes": os.path.getsize(path),
           "file_stem": os.path.splitext(os.path.basename(path))[0]}
    with h5py.File(path, "r") as f:
        a = f.attrs
        for k in a.keys():
            # 큰 사전/테이블성 attr 은 manifest 에 넣지 않는다
            if k in ("symbol_table", "aux_vocab", "fiducial_vocab",
                     "fiducial_feature_names", "quality_names", "similarity_names"):
                continue
            v = a[k]
            if isinstance(v, (bytes, str)):
                s = _to_str(v)
                row[k] = s
            elif isinstance(v, (np.ndarray, list)):
                row[k] = json.dumps(np.asarray(v).tolist())
            else:
                row[k] = v.item() if hasattr(v, "item") else v

        leads = _j(a.get("sig_name"), [])
        if isinstance(leads, list):
            row["sig_name"] = ",".join(map(str, leads))
        scale = _j(a.get("scale"), [])
        if isinstance(scale, list):
            for i, s in enumerate(scale):
                row[f"scale_{leads[i] if i < len(leads) else i}"] = float(s)

        # ---- beat 심볼 분포 (신호는 안 읽는다) ----
        if "beat/symbol" in f:
            sym = f["beat/symbol"][:]
            row["n_beats"] = int(sym.size)
            cnt = np.bincount(sym[sym != 255], minlength=len(WFDB_SYMBOLS))
            for s in KEY_SYMBOLS:
                row[f"beat_{s if s != '/' else 'paced'}"] = int(cnt[WFDB_SYMBOLS.index(s)])
            row["beat_other"] = int(cnt.sum() - sum(
                cnt[WFDB_SYMBOLS.index(s)] for s in KEY_SYMBOLS))
            row["beat_unknown"] = int((sym == 255).sum())
            dur_h = row.get("duration_h") or 1e-9
            row["mean_hr_bpm"] = round(row["n_beats"] / (dur_h * 60.0), 1)
        else:
            row["n_beats"] = 0

        # ---- 신호 품질 요약 (QC 필터용) ----
        if "seg/quality" in f:
            q = np.asarray(f["seg/quality"][:], dtype=np.float32)   # (n_seg, 5, n_sig)
            row["nan_ratio_mean"] = float(np.nanmean(q[:, 0, :]))
            row["nan_ratio_max"] = float(np.nanmax(q[:, 0, :]))
            row["amp_std_mean"] = float(np.nanmean(q[:, 2, :]))
            # 진폭이 0 에 가까운 세그먼트 = 리드 탈락/무신호 구간
            flat = (np.nan_to_num(q[:, 2, :], nan=0.0) < 1e-3).all(axis=1)
            row["flat_seg_ratio"] = float(flat.mean())

        # ---- memmap 가능 여부 (학습 성능에 직결) ----
        ds = f["signal"]
        row["contiguous"] = bool(ds.chunks is None and ds.compression is None)
        row["signal_dtype"] = str(ds.dtype)
    return row


LABEL_RE = re.compile(r"label|group|^is_|^dx_|outcome|psvt|avnrt|avrt|afib|^afl$|^aa$|^at$|^va", re.I)


def _find_near(basename, roots, max_depth=3):
    """경로가 틀렸을 때 근처(상위 2단계 + 하위 3단계)에서 같은 이름의 파일을 찾는다."""
    seen = set()
    for r in roots:
        if not r:
            continue
        a = os.path.abspath(r)
        for base in (a, os.path.dirname(a), os.path.dirname(os.path.dirname(a))):
            if base in seen or not os.path.isdir(base):
                continue
            seen.add(base)
            d0 = base.rstrip(os.sep).count(os.sep)
            for dp, dirs, names in os.walk(base):
                if dp.count(os.sep) - d0 >= max_depth:
                    dirs[:] = []
                if basename in names:
                    return os.path.join(dp, basename)
    return None


def autodetect_id(df_clin, keys_by_kind, min_rate=0.05):
    """임상 CSV 에서 record 와 가장 잘 매칭되는 (컬럼, 매칭대상) 조합을 찾는다."""
    best = None
    for col in df_clin.columns:
        vals = df_clin[col].astype(str).str.strip()
        if vals.nunique() < 2:
            continue
        for kind, keyset in keys_by_kind.items():
            rate = vals.isin(keyset).mean()
            if rate >= min_rate and (best is None or rate > best[2]):
                best = (col, kind, rate)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="v2 레코드 디렉토리")
    ap.add_argument("--out", required=True, help="출력 경로 접두사 (.csv/.parquet 자동)")
    ap.add_argument("--clinical", nargs="*", default=[],
                    help="name=path 형식. 예: psvt=/.../clinical_data_psvt.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--consistency", type=float, default=0.9,
                    help="라벨이 아닌 열을 환자 단위로 붙이는 기준: 여러 행 ID 중 값이 일정한 비율")
    ap.add_argument("--duplicates", default=None,
                    help="find_duplicates.py 결과 CSV. dup_group / dup_keep 컬럼을 붙인다")
    args = ap.parse_args()

    files = []
    for root in args.src:
        files += [e.path for e in os.scandir(root)
                  if e.is_file() and e.name.lower().endswith((".h5", ".hdf5"))]
    files.sort()
    print(f"[scan] v2 레코드 {len(files):,}개")

    t0 = time.time()
    rows, bad = [], []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor  # h5 읽기는 I/O 바운드
        with ThreadPoolExecutor(args.workers) as ex:
            for i, (p, r) in enumerate(zip(files, ex.map(_safe, files))):
                (rows if isinstance(r, dict) else bad).append(r if isinstance(r, dict)
                                                              else (p, r))
                if (i + 1) % 200 == 0:
                    print(f"\r  {i+1:,}/{len(files):,}", end="", flush=True)
    else:
        for i, p in enumerate(files):
            r = _safe(p)
            (rows if isinstance(r, dict) else bad).append(r if isinstance(r, dict) else (p, r))
    print(f"\r[scan] 완료 {len(rows):,}개 / 실패 {len(bad)}개  ({time.time()-t0:.0f}s)")
    for p, e in bad[:5]:
        print(f"  [실패] {os.path.basename(p)}: {e}")
    if not rows:
        sys.exit("읽은 레코드가 없습니다.")

    df = pd.DataFrame(rows)

    # ---- 임상 CSV 조인 ----
    keys_by_kind = {
        "record_name": set(df.get("record_name", pd.Series(dtype=str)).astype(str)),
        "file_stem": set(df["file_stem"].astype(str)),
        "pid": set(df.get("pid", pd.Series(dtype=str)).astype(str)),
    }
    for spec in args.clinical:
        if "=" not in spec:
            print(f"  [skip] --clinical 형식 오류: {spec}")
            continue
        name, rest = spec.split("=", 1)
        # name=path  또는  name=path:컬럼:대상(record_name|file_stem|pid)
        forced = None
        parts = rest.rsplit(":", 2)
        if len(parts) == 3 and parts[2] in keys_by_kind:
            path, forced = parts[0], (parts[1], parts[2])
        else:
            path = rest
        if not os.path.exists(path):
            alt = _find_near(os.path.basename(path), [os.path.dirname(path), os.getcwd()] + list(args.src))
            if alt:
                print(f"  [{name}] {path} 없음 → {alt} 사용")
                path = alt
            else:
                print(f"  [skip] 없는 파일: {path}")
                continue
        clin = pd.read_csv(path, dtype=str, keep_default_na=False)
        clin.columns = [c.lstrip("\ufeff") for c in clin.columns]      # BOM 제거
        print(f"\n[clinical:{name}] {os.path.basename(path)}  "
              f"{len(clin):,}행 × {len(clin.columns)}열")
        if forced:
            if forced[0] not in clin.columns or forced[1] not in keys_by_kind:
                print(f"  ** 지정한 조인 '{forced[0]}:{forced[1]}' 이 잘못됨. "
                      f"컬럼 {list(clin.columns)[:12]} / 대상 {list(keys_by_kind)} **")
                continue
            vals = clin[forced[0]].astype(str).str.strip()
            hit = (forced[0], forced[1], vals.isin(keys_by_kind[forced[1]]).mean())
        else:
            hit = autodetect_id(clin, keys_by_kind)
        if hit is None:
            print("  ** ID 컬럼 자동 탐지 실패 — name=path:컬럼:대상 으로 지정하세요 "
                  f"(대상: {', '.join(keys_by_kind)}) **")
            continue
        col, kind, rate = hit
        clin[col] = clin[col].astype(str).str.strip()
        n_ids = clin[col].nunique()
        print(f"  조인 키 '{col}' ↔ manifest '{kind}'  CSV 값 중 일치 {rate:.1%}  "
              f"(행 {len(clin):,} / 고유 ID {n_ids:,})")

        # 한 ID 에 행이 여러 개면(예: PID 당 record 여러 개) 첫 행만 남기면 record 별 값이
        # 엉뚱한 record 에 붙는다. ID 안에서 값이 일정한 열만 붙이고, 달라지는 열은 뺀다.
        # 한 ID 에 행이 여러 개면(예: PID 당 record 여러 개) 첫 행만 남기면 record 별 값이
        # 엉뚱한 record 에 붙는다. 열마다 판단한다.
        #   라벨로 보이는 열      : 항상 붙이고, ID 안에서 엇갈리는 ID 만 비운다
        #   그 밖의 열            : 여러 행 ID 의 대부분(--consistency)에서 일정하면 환자 단위로
        #                           보고 붙인다(엇갈린 ID 는 비움). 아니면 record 단위 값이라 뺀다.
        if n_ids < len(clin):
            others = [c for c in clin.columns if c != col]
            valid = clin[clin[col] != ""]
            g = valid.groupby(col)
            multi = int((g.size() > 1).sum())
            dropped, blanked, label_conf = [], {}, {}
            keep_cols = [col]
            for c in others:
                nun = g[c].agg(lambda x: x[x != ""].nunique())
                conflict = set(nun[nun > 1].index)
                is_label = bool(LABEL_RE.search(c))
                if is_label or len(conflict) <= (1 - args.consistency) * max(multi, 1):
                    keep_cols.append(c)
                    if conflict:
                        blanked[c] = conflict
                        if is_label:
                            label_conf[c] = len(conflict)
                else:
                    dropped.append((c, len(conflict)))
            print(f"  한 ID 에 행 여러 개: {multi:,}개 ID")
            if dropped:
                print(f"  record 마다 값이 달라 뺀 열 {len(dropped)}개 (record 단위 값은 h5 리포트 attr 참고):")
                print("    " + ", ".join(f"{c}({n}/{multi})" for c, n in dropped[:20])
                      + (" …" if len(dropped) > 20 else ""))
            if label_conf:
                print(f"  ** 경고: 같은 ID 안에서 라벨이 엇갈려 그 ID 만 비웠습니다: "
                      + ", ".join(f"{c} {n}개 ID" for c, n in label_conf.items()) + " **")
            other_blank = {c: v for c, v in blanked.items() if c not in label_conf}
            if other_blank:
                print("  일부 ID 에서만 엇갈려 그 ID 만 비운 열: "
                      + ", ".join(f"{c}({len(v)})" for c, v in other_blank.items()))
            agg = valid[keep_cols].groupby(col, as_index=False).agg(
                lambda x: next((v for v in x if v != ""), ""))
            for c, ids in blanked.items():
                agg.loc[agg[col].isin(ids), c] = ""
            clin = agg
        clin = clin.rename(columns={c: (c if c == col else f"{name}_{c}") for c in clin.columns})
        before = len(df)
        df[kind] = df[kind].astype(str)
        df = df.merge(clin, how="left", left_on=kind, right_on=col)
        if col != kind:
            df = df.drop(columns=[col])
        assert len(df) == before, "조인으로 행이 늘었습니다"
        added = [c for c in df.columns if c.startswith(f"{name}_")]
        matched = int(df[added].notna().any(axis=1).sum()) if added else 0
        print(f"  붙인 열 {len(added)}개: {', '.join(a[len(name)+1:] for a in added[:12])}"
              + (" …" if len(added) > 12 else ""))
        print(f"  manifest 측 매칭: {matched:,}/{len(df):,} record")

    # ---- 신호 중복 표시 ----
    if args.duplicates:
        dup = pd.read_csv(args.duplicates, dtype={"record": str})
        # 같은 이름이 여러 행이면(이름까지 같은 중복) 하나라도 keep 이면 keep
        dup["keep"] = dup["keep"].astype(str).str.lower().eq("true")
        dup = (dup.groupby("record", as_index=False)
                  .agg(group=("group", "first"), keep=("keep", "any"), note=("note", "first")))
        dup = dup[["record", "group", "keep", "note"]].rename(
            columns={"record": "record_name", "group": "dup_group", "keep": "dup_keep",
                     "note": "dup_note"})
        df = df.merge(dup, how="left", on="record_name")
        df["dup_keep"] = df["dup_keep"].fillna(True).astype(bool)   # 중복 아닌 것은 keep
        n_drop = int((~df["dup_keep"]).sum())
        print(f"\n[duplicates] 중복 묶음 {df['dup_group'].nunique():,}개, "
              f"제외 대상 {n_drop:,}개 → 학습에는 dup_keep==True 만 사용")

    # ---- 저장 ----
    out = args.out
    csv_path = out if out.endswith(".csv") else out + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}  ({len(df):,}행 × {len(df.columns)}열)")
    try:
        pq = csv_path[:-4] + ".parquet"
        df.to_parquet(pq, index=False)
        print(f"[saved] {pq}")
    except Exception as e:
        print(f"[parquet 생략] {type(e).__name__}: {e}  (pyarrow 미설치 시 정상)")

    # ---- 요약 ----
    print("\n" + "=" * 70)
    print("요약")
    print("=" * 70)
    if "source" in df:
        print("  코호트별:")
        for s, n in df["source"].value_counts().items():
            sub = df[df["source"] == s]
            print(f"    {s:28s} {n:>6,}개  "
                  f"{sub['duration_h'].sum():>9,.0f}h  "
                  f"beat 보유 {int(sub['has_beats'].astype(bool).sum()):,}")
    if "duration_h" in df:
        d = df["duration_h"]
        print(f"  길이(h): median {d.median():.2f}  p1 {d.quantile(.01):.2f}  "
              f"max {d.max():.2f}  총 {d.sum():,.0f}h ({d.sum()/24:,.0f} record-day)")
        short = (d < 12).sum()
        if short:
            print(f"    12시간 미만 {short:,}개 — 학습 제외 검토")
    for c in ("contiguous", "has_beats", "has_report"):
        if c in df:
            print(f"  {c}: {int(df[c].astype(bool).sum()):,}/{len(df):,}")
    if "flat_seg_ratio" in df:
        bad_q = (df["flat_seg_ratio"] > 0.2).sum()
        print(f"  무신호 세그먼트 20% 초과: {bad_q:,}개 — QC 제외 후보")
    if "pid" in df:
        print(f"  고유 환자(pid): {df['pid'].nunique():,}  "
              f"(record {len(df):,} → 환자당 {len(df)/max(df['pid'].nunique(),1):.2f})")
        print("    ※ split 은 반드시 pid 단위로 나눌 것 (같은 환자 누수 방지)")
    print(f"  총 용량: {df['file_bytes'].sum()/1e9:,.1f} GB")


def _safe(p):
    try:
        return scan_record(p)
    except Exception as e:
        return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
