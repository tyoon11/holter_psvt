# sampling.py
# -*- coding: utf-8 -*-
"""
S-우선 샘플링 (method=base 고정, fill로 전략 전환)

이 파일이 하는 일(코드로 확인 가능한 사실만):
- 입력으로 들어온 '전체 세그먼트 키(all_keys)'와 'S가 포함된 세그먼트 키(s_keys_all)'를 바탕으로,
  각 레코드에서 K(seg_per_record)개의 세그먼트를 뽑아 반환한다.
- "S 우선" 규칙:
  1) S 세그먼트를 먼저 최대한 담는다.
  2) 부족하면 S 주변(radius 반경)의 이웃 세그먼트(비S)에서 랜덤으로 보충한다.
  3) 그래도 부족하면 fill 정책(normal/duplicate/half/slike/slike_random)으로 보충한다.
- method 인자는 하위 호환용으로 존재하지만, select_keys() 내부에서는 실제로 사용하지 않는다(주석/코드로 확인).

fill 정책(코드상 구현):
- normal
    (1) 비S 중 아직 안 뽑힌 것 → (2) 전체 중 아직 안 뽑힌 것 → (3) 최후: 중복 허용 랜덤(choices)
- duplicate
    (1) S 중 아직 안 뽑힌 것 → (2) S에서 중복 허용 랜덤(choices) → (3) 전체 미선택 → (4) 전체 중복
- half
    남은 need를 S/N 절반씩 채우려 시도하고, 각 그룹에서 부족하면 중복 허용.
    그래도 남으면 전체 미선택 → 최후 전체 중복.
- slike
    CSV(score 기반)로 랭킹된 seg_index를 높은 점수부터 중복 없이 추가
    → 전체 미선택(여기선 remaining[:take]로 "앞에서부터" 채움)
    → 최후: 중복 허용 랜덤(choices)
- slike_random
    CSV(time 기반)로 얻은 "고유 seg_index 집합"에서 랜덤으로 중복 없이 추가
    → 전체 미선택(remaining[:take])
    → 최후: 중복 허용 랜덤(choices)

중요한 전제(코드상 확실):
- seg key 형식은 "seg_000123"처럼 "_" 뒤의 숫자를 인덱스로 갖는다고 가정(parse_idx).
- SLIKE CSV는 time(초) / score 컬럼을 기대하고, seg_index는 time//10으로 계산한다.
"""

from typing import List, Dict, Optional
import os
import random
import pandas as pd


# =============================================================================
# 내부 유틸: seg key <-> index
# =============================================================================
def parse_idx(k: str) -> int:
    """
    세그먼트 키 문자열에서 정수 인덱스를 추출한다.

    예) "seg_000123" -> 123

    코드가 가정하는 key 포맷:
    - "_"로 split했을 때 마지막 토큰이 정수로 변환 가능해야 한다.
    """
    return int(k.split("_")[-1])


# =============================================================================
# SLIKE CSV 경로 탐색 유틸
# =============================================================================
def _resolve_slike_csv_path(slike_dir: Optional[str], file_basename: Optional[str]) -> Optional[str]:
    """
    slike_dir / file_basename 조합으로 SLIKE CSV 파일 경로를 추정한다.

    탐색 순서(코드상):
    1) <slike_dir>/<file_basename>.csv
    2) <slike_dir>/<file_basename>_topk.csv

    추가 규칙:
    - file_basename이 "_seg_all_denoise"로 끝나면,
      그 suffix를 제거한 base에 대해서도 위 (1)(2) 패턴을 다시 시도한다.

    반환:
    - 실제로 존재하는 첫 번째 파일 경로(str)
    - 없으면 None

    안전장치(코드상):
    - slike_dir 또는 file_basename이 None/빈값이면 None
    - slike_dir이 디렉토리가 아니면 None
    """
    if not (slike_dir and file_basename):
        return None
    if not os.path.isdir(slike_dir):
        return None

    base_candidates = [file_basename]
    if file_basename.endswith("_seg_all_denoise"):
        base_candidates.append(file_basename.replace("_seg_all_denoise", ""))

    candidates: List[str] = []
    for base in base_candidates:
        candidates.append(os.path.join(slike_dir, f"{base}.csv"))
        candidates.append(os.path.join(slike_dir, f"{base}_topk.csv"))

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


# =============================================================================
# S 주변 이웃 확장 유틸
# =============================================================================
def _expand_neighbors_from_S(idx_map: Dict[int, str], s_keys_ordered: List[str], radius: int) -> List[str]:
    """
    S 세그먼트들의 인덱스를 중심으로 반경(radius)만큼의 이웃 인덱스를 모아 key 리스트로 반환한다.

    입력:
    - idx_map: {seg_index(int): seg_key(str)} (전체 키에서 만든 맵)
    - s_keys_ordered: all_keys_sorted에서 S에 해당하는 키들(순서 유지)
    - radius: S 인덱스 i에 대해 [i-radius, ..., i+radius]를 이웃으로 간주

    출력:
    - 중복 제거(set 기반)
    - 인덱스 오름차순으로 정렬한 뒤, idx_map을 통해 key로 변환한 리스트

    주의(코드상 확실):
    - radius <= 0이면 빈 리스트 반환
    - s_keys_ordered가 비어도 빈 리스트 반환
    """
    if not s_keys_ordered or radius <= 0:
        return []

    picked = set()
    indices = set(idx_map.keys())

    for k in s_keys_ordered:
        i = parse_idx(k)
        for j in range(i - radius, i + radius + 1):
            if j in indices:
                picked.add(j)

    return [idx_map[j] for j in sorted(picked)]


# =============================================================================
# SLIKE 유틸: CSV에서 seg_index 후보 읽기
# =============================================================================
def _load_ranked_indices_from_csv(slike_dir: Optional[str], file_basename: Optional[str]) -> List[int]:
    """
    SLIKE CSV에서 (time, score)를 읽어 score 내림차순으로 정렬한 뒤,
    time//10으로 seg_index를 계산하고, seg_index 중복을 제거하며 순서를 유지한다.

    반환:
    - ranked seg_index 리스트(점수 높은 순)
    - 어떤 이유로든 읽기/파싱 실패하면 빈 리스트

    구현 디테일(코드상):
    - pd.read_csv(csv_path, usecols=["time","score"])를 먼저 시도하고,
      실패하면 전체 컬럼 로드 후 time/score 컬럼 존재를 검사한다.
    - time/score는 to_numeric(errors="coerce")로 숫자화하고 NaN은 drop
    - time>=0 필터 적용
    - seg_index = (time // 10).astype(int)
    - drop_duplicates(seg_index, keep="first")로 중복 제거
    """
    csv_path = _resolve_slike_csv_path(slike_dir, file_basename)
    if csv_path is None:
        return []

    try:
        df = pd.read_csv(csv_path, usecols=["time", "score"])
    except Exception:
        df = pd.read_csv(csv_path)

    if "time" not in df.columns or "score" not in df.columns:
        return []

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["time", "score"])
    df = df[df["time"] >= 0]
    if df.empty:
        return []

    df = df.sort_values("score", ascending=False).copy()
    df["seg_index"] = (df["time"] // 10).astype(int)

    ranked = (
        df.drop_duplicates(subset=["seg_index"], keep="first")["seg_index"]
        .astype(int)
        .tolist()
    )
    return ranked


def _load_unique_indices_from_csv(slike_dir: Optional[str], file_basename: Optional[str]) -> List[int]:
    """
    SLIKE CSV에서 time 컬럼을 읽어 seg_index=time//10을 만들고,
    seg_index의 고유값 리스트를 반환한다(정렬/랭킹 보장 없음).

    반환:
    - unique seg_index 리스트
    - 실패하면 빈 리스트

    구현 디테일(코드상):
    - usecols=["time"]를 먼저 시도하고 실패하면 전체 로드
    - time 숫자화 → NaN 제거 → time>=0 필터
    - drop_duplicates로 seg_index 중복 제거
    """
    csv_path = _resolve_slike_csv_path(slike_dir, file_basename)
    if csv_path is None:
        return []

    try:
        df = pd.read_csv(csv_path, usecols=["time"])
    except Exception:
        df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        return []

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[df["time"] >= 0]
    if df.empty:
        return []

    df["seg_index"] = (df["time"] // 10).astype(int)
    return df["seg_index"].drop_duplicates().astype(int).tolist()


# =============================================================================
# fill 정책 구현
# =============================================================================
def _fill_normal(
    selected: List[str],
    all_keys: List[str],
    s_keys: List[str],
    n_keys: List[str],
    target_size: int,
    rng: random.Random
) -> List[str]:
    """
    normal fill:
    1) 비S(n_keys) 중에서 아직 선택되지 않은 것들을 랜덤 샘플
    2) 전체(all_keys) 중에서 아직 선택되지 않은 것들을 랜덤 샘플
    3) 그래도 부족하면 pool에서 중복 허용(rng.choices)으로 채움
       - pool 우선순위: n_keys가 있으면 n_keys, 없으면 all_keys

    반환 길이는 최소 target_size가 되도록 하며,
    시작에 selected가 이미 target_size 이상이면 앞에서 잘라서 반환한다.
    """
    if len(selected) >= target_size:
        return selected[:target_size]

    need = target_size - len(selected)
    selected_set = set(selected)

    # 1) 비S 미선택에서 랜덤 샘플
    n_remain = [k for k in n_keys if k not in selected_set]
    take = min(len(n_remain), need)
    if take > 0:
        selected += rng.sample(n_remain, take)

    need = target_size - len(selected)
    if need <= 0:
        return selected

    # 2) 전체 미선택에서 랜덤 샘플
    selected_set = set(selected)
    all_remain = [k for k in all_keys if k not in selected_set]
    take = min(len(all_remain), need)
    if take > 0:
        selected += rng.sample(all_remain, take)

    need = target_size - len(selected)
    if need <= 0:
        return selected

    # 3) 최후: 중복 허용
    pool = n_keys if n_keys else all_keys
    if pool and need > 0:
        selected += rng.choices(pool, k=need)

    return selected


def _fill_duplicate(
    selected: List[str],
    all_keys: List[str],
    s_keys: List[str],
    n_keys: List[str],
    target_size: int,
    rng: random.Random
) -> List[str]:
    """
    duplicate fill:
    1) S 중 아직 선택되지 않은 것들을 랜덤 샘플
    2) 그래도 부족하면 S에서 중복 허용으로 채움(rng.choices)
    3) S가 없어서 2)에서 못 채우면 전체 미선택에서 샘플
    4) 최후: 전체에서 중복 허용(rng.choices)

    코드상 특징:
    - 2)에서 S 중복으로 채울 수 있으면 즉시 return 한다.
    """
    if len(selected) >= target_size:
        return selected[:target_size]

    need = target_size - len(selected)
    selected_set = set(selected)

    # 1) S 미선택
    s_remain = [k for k in s_keys if k not in selected_set]
    take = min(len(s_remain), need)
    if take > 0:
        selected += rng.sample(s_remain, take)

    need = target_size - len(selected)
    if need <= 0:
        return selected

    # 2) S 중복 허용(가능하면 여기서 종료)
    if s_keys and need > 0:
        selected += rng.choices(s_keys, k=need)
        return selected

    # 3) 전체 미선택
    selected_set = set(selected)
    all_remain = [k for k in all_keys if k not in selected_set]
    take = min(len(all_remain), need)
    if take > 0:
        selected += rng.sample(all_remain, take)

    # 4) 최후: 전체 중복
    need = target_size - len(selected)
    if need > 0 and all_keys:
        selected += rng.choices(all_keys, k=need)

    return selected


def _fill_half(
    selected: List[str],
    all_keys: List[str],
    s_keys: List[str],
    n_keys: List[str],
    target_size: int,
    rng: random.Random
) -> List[str]:
    """
    half fill:
    - 남은 need를 절반씩 S와 N에 할당:
        s_need = ceil(need/2), n_need = floor(need/2)
    - 각 그룹에서:
        1) 미선택 pool에서 sample
        2) 부족하면 해당 그룹에서 중복 허용 choices
    - 그래도 남으면:
        3) 전체 미선택 sample
        4) 최후 전체 중복

    반환 길이는 target_size로 맞추려 시도한다.
    """
    if len(selected) >= target_size:
        return selected[:target_size]

    need = target_size - len(selected)

    # 절반 할당(need가 홀수면 S 쪽에 1 더 줌)
    s_need = (need + 1) // 2
    n_need = need - s_need

    selected_set = set(selected)

    # --- S 쪽 채우기 ---
    s_remain = [k for k in s_keys if k not in selected_set]
    take_s = min(len(s_remain), s_need)
    if take_s > 0:
        selected += rng.sample(s_remain, take_s)
    s_need -= take_s

    # 부족하면 S 중복 허용
    if s_need > 0 and s_keys:
        selected += rng.choices(s_keys, k=s_need)
        s_need = 0

    # --- N 쪽 채우기 ---
    selected_set = set(selected)
    n_remain = [k for k in n_keys if k not in selected_set]
    take_n = min(len(n_remain), n_need)
    if take_n > 0:
        selected += rng.sample(n_remain, take_n)
    n_need -= take_n

    # 부족하면 N 중복 허용
    if n_need > 0 and n_keys:
        selected += rng.choices(n_keys, k=n_need)
        n_need = 0

    # --- 잔여 처리: 전체 미선택 → 최후 중복 ---
    need_left = target_size - len(selected)
    if need_left <= 0:
        return selected

    selected_set = set(selected)
    all_remain = [k for k in all_keys if k not in selected_set]
    take = min(len(all_remain), need_left)
    if take > 0:
        selected += rng.sample(all_remain, take)

    need_left = target_size - len(selected)
    if need_left > 0 and all_keys:
        selected += rng.choices(all_keys, k=need_left)

    return selected


def _fill_slike_with_dup_tail(
    selected: List[str],
    all_keys: List[str],
    idx_map: Dict[int, str],
    file_basename: Optional[str],
    slike_dir: Optional[str],
    target_size: int,
    rng: Optional[random.Random] = None
) -> List[str]:
    """
    slike fill:
    1) CSV(score 기반)로 뽑은 ranked seg_index를 score 내림차순으로 순회하며
       idx_map에 존재하는 key를 "중복 없이" out에 append
    2) 부족하면 전체 미선택(remaining)에서 앞에서부터(remaining[:take]) 채움
       - 여기서는 rng.sample이 아니라 "정렬된 all_keys 기준 앞쪽부터" 들어간다(코드상 사실).
    3) 그래도 부족하면 pool에서 중복 허용 choices로 채움
       - pool 우선순위: ranked_keys가 있으면 ranked_keys, 없으면 all_keys, 그마저도 없으면 out

    반환은 항상 target_size로 잘라서 반환한다.
    """
    out = list(selected)
    need = target_size - len(out)
    if need <= 0:
        return out[:target_size]

    ranked_idx = _load_ranked_indices_from_csv(slike_dir, file_basename)
    ranked_keys = [idx_map[i] for i in ranked_idx if i in idx_map]

    # 1) ranked_keys를 순서대로 중복 없이 추가
    if ranked_keys:
        selected_set = set(out)
        for k in ranked_keys:
            if k not in selected_set:
                out.append(k)
                selected_set.add(k)
                if len(out) >= target_size:
                    return out[:target_size]

    # 2) 전체 미선택(remaining)에서 "앞에서부터" 채우기
    if len(out) < target_size:
        selected_set = set(out)
        remaining = [k for k in all_keys if k not in selected_set]
        take = min(len(remaining), target_size - len(out))
        if take > 0:
            out.extend(remaining[:take])

    # 3) 최후: 중복 허용
    if len(out) < target_size:
        if rng is None:
            rng = random
        pool = ranked_keys if ranked_keys else (all_keys if all_keys else out)
        out.extend(rng.choices(pool, k=target_size - len(out)))

    return out[:target_size]


def _fill_slike_random_tail(
    selected: List[str],
    all_keys: List[str],
    idx_map: Dict[int, str],
    file_basename: Optional[str],
    slike_dir: Optional[str],
    target_size: int,
    rng: random.Random
) -> List[str]:
    """
    slike_random fill:
    1) CSV(time 기반)로 얻은 uniq seg_index 집합에서,
       idx_map에 존재하는 key들을 모아 uniq_keys를 만든다.
    2) uniq_keys 중 미선택 pool에서 rng.sample로 중복 없이 채움
    3) 부족하면 전체 미선택(remaining)에서 앞에서부터(remaining[:take]) 채움
    4) 그래도 부족하면 pool에서 중복 허용 choices로 채움
       - pool 우선순위: uniq_keys → all_keys → out

    반환은 항상 target_size로 잘라서 반환한다.
    """
    out = list(selected)
    need = target_size - len(out)
    if need <= 0:
        return out[:target_size]

    uniq_idx = _load_unique_indices_from_csv(slike_dir, file_basename)
    uniq_keys = [idx_map[i] for i in uniq_idx if i in idx_map]

    # 1~2) uniq_keys에서 중복 없이 랜덤 보충
    if uniq_keys:
        selected_set = set(out)
        pool = [k for k in uniq_keys if k not in selected_set]
        take = min(len(pool), need)
        if take > 0:
            out += rng.sample(pool, take)

    need = target_size - len(out)
    if need <= 0:
        return out[:target_size]

    # 3) 전체 미선택에서 "앞에서부터" 채우기
    selected_set = set(out)
    remaining = [k for k in all_keys if k not in selected_set]
    take = min(len(remaining), need)
    if take > 0:
        out.extend(remaining[:take])

    need = target_size - len(out)
    if need <= 0:
        return out[:target_size]

    # 4) 최후: 중복 허용
    pool = uniq_keys if uniq_keys else (all_keys if all_keys else out)
    out += rng.choices(pool, k=need)

    return out[:target_size]


# =============================================================================
# 공개 API: select_keys
# =============================================================================
def select_keys(
    all_keys: List[str],
    s_keys_all: List[str],
    method: str,  # 하위 호환용; 내부에서는 사용하지 않음(코드상 fill만 보고 분기)
    radius: int,
    seg_per_record: int,
    fill: str,
    rng: random.Random,
    file_basename: Optional[str] = None,
    slike_dir: Optional[str] = None
) -> List[str]:
    """
    S-우선 샘플링의 최종 엔트리 포인트.

    입력:
    - all_keys: 해당 레코드(HDF5) 안의 전체 segment key 목록
    - s_keys_all: S 심볼이 포함된 segment key 목록(상위에서 만들어서 넘겨줌)
    - radius: S 주변 이웃으로 확장할 반경
    - seg_per_record: 최종 목표 개수 K
    - fill: 부족분을 채우는 정책 문자열
    - rng: random.Random 인스턴스(상위에서 seed 고정해서 전달)
    - file_basename / slike_dir: fill이 slike/slike_random일 때 CSV 참조에 사용

    처리 절차(코드 그대로):
    0) fill 문자열을 소문자로 정규화
    1) all_keys를 인덱스(parse_idx) 기준으로 정렬하여 all_keys_sorted 생성
       - idx_map = {index: key}
       - all_keys_sorted = [idx_map[i] for i in sorted(idx_map)]
    2) S/N 분리:
       - s_keys_ordered: all_keys_sorted 중 S set에 속하는 것(정렬 순서 유지)
       - n_keys_all: all_keys_sorted 중 S가 아닌 것
    3) S 우선:
       - S가 K개 이상이면 rng.sample로 S 중 K개를 랜덤 반환
       - 아니면 selected = S 전체로 시작
    4) 이웃 확장:
       - neighbors = _expand_neighbors_from_S(...)
       - 그 중 "비S"이며 아직 selected에 없는 것들만 candidates로 두고 rng.sample로 보충
    5) 여전히 부족하면 fill 정책으로 보충
    6) 안전하게 길이를 K로 맞춰 반환(초과하면 앞에서 자름)
    """
    # method는 무시하고 base로 처리한다는 주석/구현(실제로 method 변수는 이후 사용되지 않음)
    fill = (fill or "").lower()

    # 1) all_keys를 인덱스 기준 정렬(순서 의존성 제거 목적)
    idx_map = {parse_idx(k): k for k in all_keys}
    all_keys_sorted = [idx_map[i] for i in sorted(idx_map)]

    # 2) S set 구성 및 S/N 분리
    s_set = set(s_keys_all)
    s_keys_ordered = [k for k in all_keys_sorted if k in s_set]
    n_keys_all = [k for k in all_keys_sorted if k not in s_set]

    # 3) S가 충분하면 S 내부에서 랜덤으로 K개 선택 후 종료
    if len(s_keys_ordered) >= seg_per_record:
        return rng.sample(s_keys_ordered, seg_per_record)

    selected = list(s_keys_ordered)

    # 4) S±radius 이웃(비S)에서 랜덤 보충 (중복 없이)
    need = seg_per_record - len(selected)
    if need > 0:
        neighbors = _expand_neighbors_from_S(idx_map, s_keys_ordered, radius)
        # neighbors에는 S도 포함될 수 있으므로, 비S + 미선택 조건으로 필터링
        candidates = [k for k in neighbors if (k not in s_set) and (k not in selected)]
        take = min(len(candidates), need)
        if take > 0:
            selected += rng.sample(candidates, take)

    # 5) fill 정책으로 최종 보충
    if len(selected) < seg_per_record:
        if fill == "slike":
            selected = _fill_slike_with_dup_tail(
                selected=selected,
                all_keys=all_keys_sorted,
                idx_map=idx_map,
                file_basename=file_basename,
                slike_dir=slike_dir,
                target_size=seg_per_record,
                rng=rng,
            )
        elif fill == "slike_random":
            selected = _fill_slike_random_tail(
                selected=selected,
                all_keys=all_keys_sorted,
                idx_map=idx_map,
                file_basename=file_basename,
                slike_dir=slike_dir,
                target_size=seg_per_record,
                rng=rng,
            )
        elif fill == "duplicate":
            selected = _fill_duplicate(
                selected, all_keys_sorted, s_keys_ordered, n_keys_all, seg_per_record, rng
            )
        elif fill == "half":
            selected = _fill_half(
                selected, all_keys_sorted, s_keys_ordered, n_keys_all, seg_per_record, rng
            )
        else:
            # fill이 "normal"이거나, 그 외 알 수 없는 문자열이면 normal로 처리(코드상 else)
            selected = _fill_normal(
                selected, all_keys_sorted, s_keys_ordered, n_keys_all, seg_per_record, rng
            )

    # 6) 안전: 길이를 K로 맞춤(초과하면 앞에서 자름)
    if len(selected) > seg_per_record:
        selected = selected[:seg_per_record]

    return selected
