import os
import json
import shutil

INPUT_DIR = "leaderboard_submission"          # <-- your raw 89 files
OUTPUT_DIR = "leaderboard_submission_fixed"   # <-- new clean folder

# Create output folder fresh
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect paras key
    paras = data["body"].get("paras") or data["body"].get("paragraphs")

    # Apply Window-of-3 index increment
    for para in paras:
        mp = para.get("matched_paras", {})
        if isinstance(mp, dict):
            fixed = {}
            for k, v in mp.items():
                try:
                    fixed[str(int(k) + 1)] = v
                except:
                    continue
            para["matched_paras"] = fixed

    # Write corrected file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Window-of-3 index fix complete.")
print(f"📁 Clean files written to: {OUTPUT_DIR}")
