#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_conversion.py — 원본(.hea/.SIG/.ANN/.json) → v2 h5 대량 변환 드라이버.

convert_to_h5.py 의 convert_folder_to_h5_ray 를 수천 개 규모로 돌리기 위해 다시 짰다.

  - 여러 원본 디렉토리를 한 번에 (PSVT / TOF / LQT, 하위 디렉토리 포함)
  - CPU 제한: --cpus (기본 32). Ray 가 이 이상 코어를 잡지 않는다.
  - 제출량 제한: 동시에 떠 있는 작업을 --inflight 개로 묶는다.
      5,797 개를 한꺼번에 제출하면 스케줄러와 object store 가 불필요하게 부푼다.
  - 워커가 h5 를 직접 쓰고 요약만 돌려준다 (레코드당 130MB 왕복 제거).
  - 재개: 출력에 이미 있는 record 는 건너뛴다. v2 writer 가 .tmp → 원자적 교체라
    파일이 존재하면 완성본이다. 중단으로 남은 .tmp 는 시작 시 정리한다.
  - 진행도: 완료/전체, 성공/실패, 속도, 경과, ETA, 쓴 용량
  - 레코드별 결과를 CSV 에 즉시 append → 중단돼도 기록이 남는다.

사용:
  python tools/run_conversion.py \\
      --raw /home/coder/workspace/Holter_TOF/nas1_Holter_PSVT \\
            /home/coder/workspace/Holter_TOF/nas1_Holter_TOF_250917 \\
            /home/coder/workspace/Holter_TOF/nas1_Holter_LQT_260615 \\
      --out /home/coder/workspace/data/holter_v2 --cpus 32

  python tools/run_conversion.py ... --limit 3 --cpus 3      # 시험
  python tools/run_conversion.py ... --dry-run               # 대상만 집계
  python tools/run_conversion.py ... --real-fiducial         # neurokit fiducial 추출 (느림)
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = os.path.join(HERE, "..", "h5_converter")

# utils.has_all_required_files 와 같은 조건 (대소문자 포함)
REQUIRED = (".hea", ".SIG", ".ANN", ".json")


# =============================================================================
# 대상 수집
# =============================================================================
def gather(raw_dirs):
    """(record_name, hea_path) 목록과 제외 사유 집계를 만든다."""
    found, skipped, dup = {}, Counter(), []
    for root in raw_dirs:
        if not os.path.isdir(root):
            print(f"  [skip] 없는 디렉토리: {root}")
            continue
        for dirpath, _, names in os.walk(root):
            nameset = set(names)
            for nm in names:
                if not nm.lower().endswith(".hea"):
                    continue
                base = nm[:-4]
                missing = [e for e in REQUIRED[1:] if base + e not in nameset]
                if missing:
                    for e in missing:
                        skipped[f"{e} 없음"] += 1
                    continue
                if base in found:
                    dup.append((base, found[base], os.path.join(dirpath, nm)))
                    continue
                found[base] = os.path.join(dirpath, nm)
    return found, skipped, dup


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024.0


def hms(sec):
    if sec != sec or sec == float("inf") or sec < 0:
        return "--:--:--"
    sec = int(sec)
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# =============================================================================
# 실행 백엔드 — ray 기본, process 는 ray 없이 돌리거나 시험할 때
# =============================================================================
def _pythonpath():
    """워커가 convert_to_h5 / utils / schema_v2 를 import 할 수 있는 PYTHONPATH.

    convert_to_h5.py 는 'from utils import ...' 처럼 평면 import 를 쓰므로
    h5_converter 디렉토리 자체가 경로에 있어야 한다. 드라이버의 sys.path 에만 넣으면
    Ray 워커는 상속받지 못해 ModuleNotFoundError 로 전부 실패한다.
    """
    conv = os.path.abspath(CONV_DIR)
    old = os.environ.get("PYTHONPATH", "")
    return conv + (os.pathsep + old if old else "")


class RayBackend:
    def __init__(self, cpus, tmpdir=None):
        import ray
        self.ray = ray
        kw = dict(num_cpus=cpus, include_dashboard=False, ignore_reinit_error=True,
                  log_to_driver=False,
                  runtime_env={"env_vars": {"PYTHONPATH": _pythonpath()}})
        if tmpdir:
            kw["_temp_dir"] = tmpdir
        ray.init(**kw)
        self._fn = None

    def submit(self, fn, *args, **kw):
        if self._fn is None:
            self._fn = self.ray.remote(num_cpus=1)(fn)
        return self._fn.remote(*args, **kw)

    def wait(self, handles):
        done, pending = self.ray.wait(list(handles), num_returns=1, timeout=1.0)
        return done, pending

    def get(self, h):
        return self.ray.get(h)

    def close(self):
        self.ray.shutdown()


class ProcessBackend:
    def __init__(self, cpus, tmpdir=None):
        from concurrent.futures import ProcessPoolExecutor
        self.ex = ProcessPoolExecutor(max_workers=cpus)

    def submit(self, fn, *args, **kw):
        return self.ex.submit(fn, *args, **kw)

    def wait(self, handles):
        from concurrent.futures import wait, FIRST_COMPLETED
        done, pending = wait(list(handles), timeout=1.0, return_when=FIRST_COMPLETED)
        return list(done), list(pending)

    def get(self, h):
        return h.result()

    def close(self):
        self.ex.shutdown(cancel_futures=True)


# =============================================================================
# 진행도
# =============================================================================
class Progress:
    def __init__(self, total):
        self.total, self.ok, self.err, self.bytes = total, 0, 0, 0
        self.t0 = time.time()
        self.last = 0.0
        self.tty = sys.stdout.isatty()
        self.last_line_print = 0.0

    def update(self, res, force=False):
        if res is not None:
            if res.get("status") == "ok":
                self.ok += 1
                self.bytes += res.get("bytes", 0)
            else:
                self.err += 1
        now = time.time()
        # tty 는 0.5초마다 덮어쓰기, 로그 파일로 갈 때는 30초마다 한 줄
        gap = 0.5 if self.tty else 30.0
        if not force and now - self.last < gap:
            return
        self.last = now
        done = self.ok + self.err
        el = now - self.t0
        rate = done / el if el > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        frac = done / self.total if self.total else 1.0
        w = 28
        bar = "█" * int(w * frac) + "·" * (w - int(w * frac))
        msg = (f"  [{bar}] {frac*100:5.1f}%  {done:,}/{self.total:,}  "
               f"성공 {self.ok:,} 실패 {self.err:,}  {rate*60:.1f} rec/min  "
               f"경과 {hms(el)}  ETA {hms(eta)}  출력 {human(self.bytes)}")
        if self.tty:
            print("\r" + msg + "   ", end="", flush=True)
        else:
            print(msg, flush=True)

    def finish(self):
        self.update(None, force=True)
        if self.tty:
            print()


# =============================================================================
def run(records, out_dir, backend, task_fn, task_kw, inflight, log_path):
    """제출량을 inflight 로 묶어 돌린다. records: [(name, hea_path)]"""
    prog = Progress(len(records))
    log_new = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    cols = ["time", "record_name", "status", "n_seg", "leads", "src_leads", "bytes", "sec",
            "error", "hea"]
    w = csv.DictWriter(logf, fieldnames=cols, extrasaction="ignore")
    if log_new:
        w.writeheader()

    it = iter(records)
    pending = {}                  # handle -> (name, hea, t_submit)
    errors = []
    try:
        while True:
            while len(pending) < inflight:
                try:
                    name, hea = next(it)
                except StopIteration:
                    break
                h = backend.submit(task_fn, hea, output_dir=out_dir, **task_kw)
                pending[h] = (name, hea, time.time())
            if not pending:
                break
            done, _ = backend.wait(pending.keys())
            for h in done:
                name, hea, t_sub = pending.pop(h)
                try:
                    res = backend.get(h)
                except Exception as e:     # 워커 프로세스 자체가 죽은 경우
                    res = {"record_name": name, "status": "error",
                           "error": f"{type(e).__name__}: {e}"}
                if res is None:
                    res = {"record_name": name, "status": "error", "error": "None 반환"}
                res.setdefault("record_name", name)
                res["sec"] = round(time.time() - t_sub, 1)
                res["hea"] = hea
                res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                w.writerow(res)
                logf.flush()
                if res.get("status") != "ok":
                    errors.append(res)
                prog.update(res)
            prog.update(None)
    except KeyboardInterrupt:
        prog.finish()
        print("\n중단했습니다. 같은 명령을 다시 실행하면 완료분을 건너뛰고 이어갑니다.")
        logf.close()
        raise
    prog.finish()
    logf.close()
    return prog, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True, help="원본 디렉토리 (재귀 탐색)")
    ap.add_argument("--out", required=True, help="v2 h5 출력 디렉토리")
    ap.add_argument("--cpus", type=int, default=32, help="사용할 CPU 코어 수 (기본 32)")
    ap.add_argument("--inflight", type=int, default=None,
                    help="동시에 제출해 둘 작업 수 (기본 cpus*2)")
    ap.add_argument("--backend", choices=["ray", "process"], default="ray")
    ap.add_argument("--ray-tmp", default=None, help="Ray 임시 디렉토리 (/tmp 가 작을 때)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--real-fiducial", action="store_true",
                    help="neurokit fiducial 추출 수행 (기본은 dummy, 훨씬 빠름)")
    ap.add_argument("--real-similarity", action="store_true",
                    help="beat 유사도(corr/DTW) 계산 수행 (기본은 dummy)")
    ap.add_argument("--log", default=None, help="결과 CSV (기본 <out>/conversion_log.csv)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = args.log or os.path.join(args.out, "conversion_log.csv")

    print("[1/3] 대상 수집")
    found, skipped, dup = gather(args.raw)
    print(f"  변환 가능(.hea+.SIG+.ANN+.json) {len(found):,}개")
    if skipped:
        print("  제외: " + "  ".join(f"{k} {v:,}" for k, v in skipped.most_common()))
    if dup:
        print(f"  ** 같은 record 이름이 여러 곳에 있음 {len(dup):,}건 — 먼저 찾은 것만 사용 **")
        for b, a1, a2 in dup[:3]:
            print(f"     {b}: {a1}  |  {a2}")

    # 중단 잔재 정리 + 재개
    stale = 0
    for e in os.scandir(args.out):
        if e.name.endswith(".h5.tmp"):
            try:
                os.remove(e.path); stale += 1
            except OSError:
                pass
    existing = {e.name[:-3] for e in os.scandir(args.out) if e.name.endswith(".h5")}
    todo = sorted((n, p) for n, p in found.items() if n not in existing)
    if args.limit:
        todo = todo[: args.limit]
    print(f"  이미 변환됨 {len(found) - len([n for n in found if n not in existing]):,}개"
          f" → 이번 작업 {len(todo):,}개")
    if stale:
        print(f"  이전 중단으로 남은 .tmp {stale}개 정리")

    try:
        st = os.statvfs(args.out)
        free = st.f_bavail * st.f_frsize
        need = len(todo) * 65e6
        print(f"  출력 여유공간 {human(free)} / 예상 필요 {human(need)} (record당 ~65MB)")
        if free < need * 1.05:
            print("  ** 경고: 여유공간이 부족할 수 있습니다 **")
    except (OSError, AttributeError):
        pass

    if args.dry_run or not todo:
        print("\n[dry-run] 변환하지 않았습니다." if args.dry_run else "\n할 작업이 없습니다.")
        return

    inflight = args.inflight or args.cpus * 2
    print(f"\n[2/3] 변환  backend={args.backend}  cpus={args.cpus}  inflight={inflight}  "
          f"fiducial={'real' if args.real_fiducial else 'dummy'}  "
          f"similarity={'real' if args.real_similarity else 'dummy'}")
    print(f"  로그 {log_path}")

    sys.path.insert(0, os.path.abspath(CONV_DIR))
    os.environ["PYTHONPATH"] = _pythonpath()     # process 백엔드 자식 프로세스용
    from convert_to_h5 import convert_one_record  # noqa: E402

    task_kw = dict(sampling_rate=125, segment_sec=10, max_segments=None,
                   use_dummy_fiducial=not args.real_fiducial,
                   use_dummy_similarity=not args.real_similarity)

    Backend = RayBackend if args.backend == "ray" else ProcessBackend
    backend = Backend(args.cpus, args.ray_tmp)
    t0 = time.time()
    try:
        prog, errors = run(todo, args.out, backend, convert_one_record, task_kw,
                           inflight, log_path)
    finally:
        backend.close()

    el = time.time() - t0
    print(f"\n[3/3] 완료  {hms(el)}  성공 {prog.ok:,}  실패 {prog.err:,}  출력 {human(prog.bytes)}")
    if prog.ok:
        print(f"  record당 평균 {el / max(prog.ok + prog.err, 1):.1f}초 (벽시계 기준)")
    if errors:
        reasons = Counter(str(e.get("error", ""))[:60] for e in errors)
        print("  실패 사유 상위:")
        for r, n in reasons.most_common(5):
            print(f"    {n:>5,}  {r}")
        print(f"  전체 목록은 {log_path} 에서 status=error 로 필터")


if __name__ == "__main__":
    main()
