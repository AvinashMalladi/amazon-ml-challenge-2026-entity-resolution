"""
Packaging script for Amazon ML Challenge 2026 Submission.
Validates the output files and creates the final <team_name>_submission.zip package.
"""

import os
import sys
import subprocess
import zipfile

TEAM_NAME = "CodeCanvas"
ZIP_NAME = f"{TEAM_NAME}_submission.zip"

print("=" * 70)
print(f"Validating and Packaging Submission for Team: {TEAM_NAME}")
print("=" * 70)

# Step 1: Run validation script
print("\n--- Running Official Validator ---")
cmd = [
    sys.executable,
    "student_resource/utils/validate_submission.py",
    "--matching", "output/matching_results.tsv",
    "--candidate", "output/candidate_pairs.tsv",
    "--test-dir", "student_resource/dataset/test"
]

res = subprocess.run(cmd)
if res.returncode != 0:
    print(f"\n[ERROR] Submission validation failed with code {res.returncode}. Aborting zip creation.")
    sys.exit(res.returncode)

print("\n[SUCCESS] Validation PASSED with exit code 0!")

# Step 2: Create submission zip archive
print(f"\n--- Creating Final Submission Package: {ZIP_NAME} ---")
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
    # 1. output/ folder
    print("Adding output/matching_results.tsv...")
    z.write("output/matching_results.tsv", "output/matching_results.tsv")
    print("Adding output/candidate_pairs.tsv...")
    z.write("output/candidate_pairs.tsv", "output/candidate_pairs.tsv")
    
    # 2. code/ folder
    code_dir = "code/business_entity_resolution"
    for root, dirs, files in os.walk(code_dir):
        for f in files:
            full_path = os.path.join(root, f)
            arc_name = full_path.replace("\\", "/")
            print(f"Adding {arc_name}...")
            z.write(full_path, arc_name)
            
    # 3. Documentation_template.md
    print("Adding Documentation_template.md...")
    z.write("Documentation_template.md", "Documentation_template.md")

zip_size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
print(f"\n[COMPLETE] Successfully created {ZIP_NAME} ({zip_size_mb:.2f} MB)!")
print("=" * 70)
