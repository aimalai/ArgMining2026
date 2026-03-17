import json
import os
import zipfile
import shutil

# --- CONFIGURATION ---
folder_path = r'C:\Users\Samsa\.vscode\coding\ArgMining2026\index_correction'
input_zip = 'ockham.zip'
output_zip = 'ockham_fixed_v7.zip'
temp_dir = os.path.join(folder_path, 'temp_processing')

def repair_bulk_indices():
    zip_path = os.path.join(folder_path, input_zip)
    final_output_zip = os.path.join(folder_path, output_zip)

    if not os.path.exists(zip_path):
        print(f"❌ Error: Could not find {input_zip} in {folder_path}")
        return

    # 1. Prepare clean temporary workspace
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"📦 Extracting {input_zip}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # 2. Loop through all extracted JSON files
    json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
    print(f"🔄 Processing {len(json_files)} files...")

    for filename in json_files:
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Access paragraphs (Standard schema: body -> paras)
        paras = data.get('body', {}).get('paras', [])
        
        for para in paras:
            rel = para.get('matched_paras', {})
            fixed_rel = {}
            current_num = para.get('para_number', 0)

            if isinstance(rel, dict):
                for k, v in rel.items():
                    try:
                        # APPLY THE FIXES:
                        # 1. Shift 0-based to 1-based
                        new_idx_val = int(k) + 1
                        
                        # 2. Strict Window-of-3 Filter
                        if new_idx_val >= (current_num - 3) and new_idx_val < current_num:
                            fixed_rel[str(new_idx_val)] = v
                    except:
                        continue
            
            para['matched_paras'] = fixed_rel

        # Save the fixed JSON back to temp folder
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # 3. Zip everything back up
    print(f"🎁 Packaging into {output_zip}...")
    shutil.make_archive(final_output_zip.replace('.zip', ''), 'zip', temp_dir)

    # 4. Cleanup temp folder
    shutil.rmtree(temp_dir)
    print(f"✅ DONE! Your repaired submission is at: {final_output_zip}")

if __name__ == "__main__":
    repair_bulk_indices()