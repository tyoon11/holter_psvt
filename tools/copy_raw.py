#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_raw.py — 원본 데이터를 로컬로 복사한다. 진행도 표시, 재개 가능, 검증 포함.

NAS 읽기가 86 MB/s 라 변환도 학습도 I/O 에 묶인다. 한 번 로컬로 옮겨두면
변환이 빨라지고 이후 학습 읽기도 살아난다. 다만 수천 개 파일 / 수백 GB 를
옮기는 동안 중단될 수 있으므로 다음을 지킨다.

  - .part 로 받은 뒤 os.replace 로 원자적 교체 → 중단되어도 깨진 파일이 남지 않음
  - 목적지에 같은 크기의 파일이 이미 있으면 건너뜀 → 재실행하면 이어서 진행
  - 진행도: 파일 수 / 용량 / 속도 / ETA 를 한 줄로 갱신
  - 원본은 읽기만 한다 (--move 같은 옵션은 두지 않았다)

사용:
  # plan_migration.py 가 만든 목록으로
  python copy_raw.py --files-from migration_plan_files.txt --base /nas/raw \
                     --dest /home/coder/workspace/data/raw --workers 8

  # 디렉토리 통째로 (확장자 필터 가능)
  python copy_raw.py --src /nas/holter_raw --dest /home/coder/workspace/data/raw \
                     --ext .dat .hea .json --workers 8

  # 변환에 필요한 원본만: .hea+.SIG 짝이 맞는 record 의 .hea/.SIG/.ANN/.json
  python copy_raw.py --src /home/coder/workspace/Holter_TOF/nas1_Holter_PSVT ... \
                     --dest /home/coder/workspace/data/raw --records-only --workers 8

  python copy_raw.py ... --dry-run     # 용량/파일수만 계산
  python copy_raw.py ... --verify      # 복사 후 크기 + 해시 확인 (느림)
"""

import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

BUF = 8 * 1024 * 1024        # 8MB — NAS 에서는 큰 버퍼가 유리하다


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024.0


def hms(sec):
    if sec != sec or sec in (float("inf"), float("-inf")) or sec < 0:
        return "--:--"
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Progress:
    """한 줄 갱신 진행도. 파이프로 넘길 때는 주기적으로 줄바꿈 출력한다."""

    def __init__(self, n_files, n_bytes, interval=0.3):
        self.n_files, self.n_bytes = n_files, n_bytes
        self.done_files = self.done_bytes = self.skipped = self.failed = 0
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.last = 0.0
        self.interval = interval
        self.tty = sys.stdout.isatty()

    def add(self, nbytes, skipped=False, failed=False):
        with self.lock:
            self.done_files += 1
            self.done_bytes += nbytes
            self.skipped += bool(skipped)
            self.failed += bool(failed)
            now = time.time()
            if now - self.last >= self.interval or self.done_files == self.n_files:
                self.last = now
                self._render()

    def _render(self):
        el = max(time.time() - self.t0, 1e-9)
        frac = self.done_bytes / self.n_bytes if self.n_bytes else 1.0
        rate = self.done_bytes / el
        eta = (self.n_bytes - self.done_bytes) / rate if rate > 0 else float("inf")
        width = 28
        fill = int(width * min(frac, 1.0))
        bar = "█" * fill + "·" * (width - fill)
        msg = (f"  [{bar}] {frac*100:5.1f}%  "
               f"{self.done_files:,}/{self.n_files:,} 파일  "
               f"{human(self.done_bytes)}/{human(self.n_bytes)}  "
               f"{human(rate)}/s  경과 {hms(el)}  ETA {hms(eta)}")
        if self.skipped:
            msg += f"  건너뜀 {self.skipped:,}"
        if self.failed:
            msg += f"  실패 {self.failed:,}"
        if self.tty:
            print("\r" + msg + " " * 6, end="", flush=True)
        else:
            print(msg, flush=True)

    def finish(self):
        self._render()
        if self.tty:
            print()


def sha256(path, chunk=BUF):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_one(src, dst, size, verify=False, overwrite=False):
    """반환: (상태, 복사바이트, 메시지)"""
    try:
        if os.path.exists(dst) and not overwrite:
            if os.path.getsize(dst) == size:
                return "skip", size, ""
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        part = dst + ".part"
        with open(src, "rb") as fi, open(part, "wb") as fo:
            shutil.copyfileobj(fi, fo, BUF)
            fo.flush()
            os.fsync(fo.fileno())
        if os.path.getsize(part) != size:
            os.remove(part)
            return "error", 0, f"크기 불일치 {os.path.getsize(part)} != {size}"
        if verify and sha256(part) != sha256(src):
            os.remove(part)
            return "error", 0, "해시 불일치"
        os.replace(part, dst)
        shutil.copystat(src, dst, follow_symlinks=False)
        return "ok", size, ""
    except Exception as e:
        for junk in (dst + ".part",):
            if os.path.exists(junk):
                try:
                    os.remove(junk)
                except OSError:
                    pass
        return "error", 0, f"{type(e).__name__}: {e}"


def gather(args):
    """복사할 (src, dst, size) 목록을 만든다."""
    items = []
    if args.files_from:
        if not args.base:
            sys.exit("--files-from 을 쓰면 --base (목록의 기준 경로)도 필요합니다.")
        with open(args.files_from) as f:
            rels = [l.strip() for l in f if l.strip()]
        for rel in rels:
            s = os.path.join(args.base, rel)
            try:
                sz = os.path.getsize(s)
            except OSError as e:
                print(f"  [없음] {s}: {e}")
                continue
            items.append((s, os.path.join(args.dest, rel), sz))
    elif args.records_only:
        # 같은 디렉토리에서 <stem>.hea 와 <stem>.SIG 가 둘 다 있는 record 만 고르고,
        # 그 record 의 .hea/.SIG/.ANN/.json 을 복사한다. 신호 없는 .json/.ann 고아나
        # 하위의 h5/denoised 디렉토리는 자연히 빠진다.
        wanted = (".hea", ".SIG", ".ANN", ".json")
        stats = Counter()
        for root in args.src:
            root = os.path.abspath(root)
            for dirpath, _, names in os.walk(root):
                nameset = set(names)
                for nm in names:
                    if not nm.endswith(".hea"):
                        continue
                    stem = nm[:-4]
                    if stem + ".SIG" not in nameset:
                        stats["신호(.SIG) 없는 .hea — 제외"] += 1
                        continue
                    stats["record"] += 1
                    for ext in wanted:
                        fn = stem + ext
                        if fn not in nameset:
                            stats[f"{ext} 없음"] += 1
                            continue
                        s_ = os.path.join(dirpath, fn)
                        rel = fn if args.flatten else os.path.relpath(s_, os.path.dirname(root))
                        try:
                            items.append((s_, os.path.join(args.dest, rel), os.path.getsize(s_)))
                        except OSError:
                            pass
        print(f"  record(.hea+.SIG) {stats.pop('record', 0):,}개"
              + "".join(f"\n    {k}: {v:,}" for k, v in stats.most_common()))
    else:
        exts = tuple(e.lower() if e.startswith(".") else "." + e.lower()
                     for e in (args.ext or []))
        for root in args.src:
            root = os.path.abspath(root)
            for dirpath, _, names in os.walk(root):
                for nm in names:
                    if exts and not nm.lower().endswith(exts):
                        continue
                    s = os.path.join(dirpath, nm)
                    try:
                        sz = os.path.getsize(s)
                    except OSError:
                        continue
                    rel = (nm if args.flatten
                           else os.path.relpath(s, os.path.dirname(root)))
                    items.append((s, os.path.join(args.dest, rel), sz))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=[], help="복사할 원본 디렉토리")
    ap.add_argument("--files-from", help="plan_migration.py 가 만든 목록 파일")
    ap.add_argument("--base", help="--files-from 목록의 기준 경로")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--ext", nargs="*", default=None,
                    help="확장자 필터, 예: --ext .dat .hea .json")
    ap.add_argument("--workers", type=int, default=8,
                    help="병렬 스트림 수. NAS 는 스트림을 늘리면 총 대역폭이 올라간다")
    ap.add_argument("--records-only", action="store_true",
                    help=".hea+.SIG 짝이 맞는 record 의 .hea/.SIG/.ANN/.json 만 복사")
    ap.add_argument("--flatten", action="store_true",
                    help="하위 디렉토리 구조를 버리고 dest 바로 아래에 둔다")
    ap.add_argument("--verify", action="store_true", help="복사 후 sha256 비교 (느림)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None, help="실패 목록 CSV")
    args = ap.parse_args()

    if not args.src and not args.files_from:
        sys.exit("--src 또는 --files-from 중 하나는 필요합니다.")

    print("[1/3] 목록 작성 중...")
    items = gather(args)
    if not items:
        sys.exit("복사할 파일이 없습니다.")
    total_bytes = sum(sz for _, _, sz in items)

    # 이미 있는 것 제외해 실제 작업량을 먼저 보여준다.
    # 이전 실행이 중단되며 남은 .part 도 같이 정리한다 (안 하면 영구히 남는다).
    todo, already, stale = [], 0, 0
    for s, d, sz in items:
        if not args.overwrite and os.path.exists(d) and os.path.getsize(d) == sz:
            already += 1
            part = d + ".part"
            if os.path.exists(part):
                try:
                    os.remove(part)
                    stale += 1
                except OSError:
                    pass
            continue
        todo.append((s, d, sz))
    todo_bytes = sum(sz for _, _, sz in todo)

    print(f"  대상 {len(items):,} 파일  {human(total_bytes)}")
    print(f"  이미 완료 {already:,} 파일 → 남은 작업 {len(todo):,} 파일  {human(todo_bytes)}")
    if stale:
        print(f"  이전 중단으로 남아있던 .part {stale:,}개 정리함")
    print(f"  목적지 {args.dest}")

    # 여유 공간 확인
    try:
        os.makedirs(args.dest, exist_ok=True)
        st = os.statvfs(args.dest)
        free = st.f_bavail * st.f_frsize
        print(f"  목적지 여유공간 {human(free)}")
        if free < todo_bytes * 1.05:
            print(f"  ** 경고: 여유공간이 부족합니다 (필요 {human(todo_bytes)}) **")
            if not args.dry_run:
                sys.exit("중단합니다. --dest 를 바꾸거나 공간을 확보하세요.")
    except (OSError, AttributeError):
        pass

    if args.dry_run:
        print("\n[dry-run] 실제 복사는 하지 않았습니다.")
        for s, d, sz in todo[:5]:
            print(f"    {s}\n      → {d}  ({human(sz)})")
        if len(todo) > 5:
            print(f"    ... 외 {len(todo)-5:,}개")
        return

    if not todo:
        print("\n모든 파일이 이미 복사되어 있습니다.")
        return

    print(f"\n[2/3] 복사 ({args.workers} 스트림)"
          f"{'  + 해시 검증' if args.verify else ''}")
    prog = Progress(len(todo), todo_bytes)
    errors = []
    t0 = time.time()
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(copy_one, s, d, sz, args.verify, args.overwrite): (s, d, sz)
                    for s, d, sz in todo}
            for fut in as_completed(futs):
                s, d, sz = futs[fut]
                status, nb, msg = fut.result()
                if status == "error":
                    errors.append((s, d, msg))
                prog.add(sz if status != "error" else 0,
                         skipped=(status == "skip"), failed=(status == "error"))
    except KeyboardInterrupt:
        prog.finish()
        print("\n중단되었습니다. 다시 실행하면 이어서 진행합니다.")
        sys.exit(130)
    prog.finish()

    el = time.time() - t0
    print(f"\n[3/3] 완료  {human(prog.done_bytes)} / {hms(el)} "
          f"(평균 {human(prog.done_bytes/max(el,1e-9))}/s)")
    print(f"  성공 {prog.done_files - prog.skipped - prog.failed:,}  "
          f"건너뜀 {prog.skipped:,}  실패 {prog.failed:,}")
    if errors:
        print("  실패 목록 (앞 10개):")
        for s, _, m in errors[:10]:
            print(f"    {os.path.basename(s)}: {m}")
        if args.log:
            import csv
            with open(args.log, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["src", "dst", "error"]); w.writerows(errors)
            print(f"  [log] {args.log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
