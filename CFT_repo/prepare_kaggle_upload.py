"""
Kaggle 업로드용 파일 2개를 준비합니다.

1. CFT_repo.zip  — 코드 전체 (runs/, *.cache 제외)
2. M3FD_yolo.zip — 데이터셋 (C:/CFT_M3FD/data/M3FD_yolo/)

실행:  python prepare_kaggle_upload.py
출력:  D:/kaggle_upload/CFT_repo.zip
       D:/kaggle_upload/M3FD_yolo.zip
"""

import os, zipfile, shutil
from pathlib import Path

REPO_DIR  = Path(__file__).parent          # CFT_repo/
M3FD_DIR  = Path('C:/CFT_M3FD/data/M3FD_yolo')
OUT_DIR   = Path('D:/kaggle_upload')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. CFT_repo.zip ───────────────────────────────────────────
EXCLUDE_DIRS  = {'runs', '__pycache__', '.git', 'wandb'}
EXCLUDE_EXT   = {'.cache', '.pyc', '.pth'}

repo_zip = OUT_DIR / 'CFT_repo.zip'
print(f'Packing {REPO_DIR} → {repo_zip}')
with zipfile.ZipFile(repo_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in REPO_DIR.rglob('*'):
        if any(ex in path.parts for ex in EXCLUDE_DIRS):
            continue
        if path.suffix in EXCLUDE_EXT:
            continue
        if path.is_file():
            arcname = 'CFT_repo/' + str(path.relative_to(REPO_DIR))
            zf.write(path, arcname)
print(f'  Done: {repo_zip.stat().st_size / 1e6:.1f} MB')

# ── 2. M3FD_yolo.zip ──────────────────────────────────────────
if M3FD_DIR.exists():
    m3fd_zip = OUT_DIR / 'M3FD_yolo.zip'
    print(f'Packing {M3FD_DIR} → {m3fd_zip}')
    with zipfile.ZipFile(m3fd_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in M3FD_DIR.rglob('*'):
            if path.is_file():
                arcname = str(path.relative_to(M3FD_DIR))
                zf.write(path, arcname)
    print(f'  Done: {m3fd_zip.stat().st_size / 1e6:.1f} MB')
else:
    print(f'[SKIP] M3FD_yolo not found at {M3FD_DIR}')
    print('  → run prepare_m3fd.py first to generate the dataset')

print('\n=== Upload checklist ===')
print('1. Go to https://www.kaggle.com/datasets')
print('2. "New Dataset" → upload M3FD_yolo.zip → name: m3fd-yolo')
print('3. "New Dataset" → upload CFT_repo.zip  → name: cft-repo')
print('4. Create new notebook → "Add Data" → add both datasets')
print('5. Upload kaggle_train_cmafm.ipynb as the notebook')
print('6. Set accelerator: GPU P100 or T4 x2 or A100')
print('7. Run all cells')
