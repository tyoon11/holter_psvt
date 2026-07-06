import os
import json
import stat
from tqdm import tqdm


def fix_hea_and_json_by_filename(input_dir):
    """
    ✅ .hea 파일:
        - 1행: record_name을 파일명으로 변경
        - 2행~: 'XXX.SIG' 부분의 파일명 앞부분만 교체 (확장자 .SIG 유지)
    ✅ .json 파일:
        - PatientInfo["PID"] 값을 파일명에서 추출한 PID로 교체
    """
    files = os.listdir(input_dir)
    base_names = sorted(set(os.path.splitext(f)[0] for f in files))

    for base_name in tqdm(base_names, desc="🛠 PID 및 record_name 정정 중", ncols=100):
        record_name = base_name  # ex: 10_50_2247355
        pid = base_name.split("_")[-1]  # ex: 2247355

        hea_path = os.path.join(input_dir, base_name + ".hea")
        json_path = os.path.join(input_dir, base_name + ".json")

        # ───────────────
        # ✅ 1. .hea 수정
        # ───────────────
        if os.path.exists(hea_path):
            try:
                with open(hea_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_lines = []
                for idx, line in enumerate(lines):
                    parts = line.strip().split()

                    if not parts:
                        new_lines.append(line)
                        continue

                    if idx == 0:
                        # 첫 줄: record_name 전체 수정
                        parts[0] = record_name
                    else:
                        # 이후 줄들: 'XXX.SIG' 중 'XXX' 부분만 수정
                        sig_parts = parts[0].split(".")
                        if len(sig_parts) == 2 and sig_parts[1].upper() == "SIG":
                            parts[0] = f"{record_name}.SIG"

                    new_lines.append(" ".join(parts) + "\n")

                try:
                    with open(hea_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                except PermissionError:
                    os.chmod(hea_path, stat.S_IWUSR | stat.S_IRUSR)
                    with open(hea_path, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)

            except Exception:
                continue  # 예외 무시하고 다음으로

        # ────────────────
        # ✅ 2. .json 수정
        # ────────────────
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "Holter Report" in data:
                    data["Holter Report"].setdefault("PatientInfo", {})
                    data["Holter Report"]["PatientInfo"]["PID"] = pid

                    try:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    except PermissionError:
                        os.chmod(json_path, stat.S_IWUSR | stat.S_IRUSR)
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

            except Exception:
                continue


# ✅ 실행 예시
if __name__ == "__main__":
    fix_hea_and_json_by_filename("/your/raw/path")