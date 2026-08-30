"""Install supplied-checkpoint class aliases into the isolated vendored engine."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


COMMON_MARKER = "# CMAFM_CHECKPOINT_COMPAT"
COMMON_IMPORT = (
    f"\n{COMMON_MARKER}\n"
    "from .cmafm_checkpoint import CMAFM_Fusion, _CMAFM\n"
)

NEW_PARSER_BLOCK = """        elif m is CMAFM_Fusion:
            c1 = [ch[x] for x in f]
            c2 = args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2]"""

CHECKPOINT_PARSER_BLOCK = """        elif m is CMAFM_Fusion:
            c2 = ch[f[0]]
            args = [c2]"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--compat-source", type=Path, required=True)
    args = parser.parse_args()

    engine_dir = args.engine_dir.resolve()
    models_dir = engine_dir / "models"
    common_path = models_dir / "common.py"
    parser_path = models_dir / "yolo_test.py"
    if not common_path.is_file() or not parser_path.is_file():
        raise SystemExit(f"Vendored engine is incomplete: {engine_dir}")

    shutil.copy2(args.compat_source, models_dir / "cmafm_checkpoint.py")

    common = common_path.read_text(encoding="utf-8")
    if COMMON_MARKER not in common:
        common_path.write_text(common.rstrip() + COMMON_IMPORT, encoding="utf-8")

    yolo = parser_path.read_text(encoding="utf-8")
    yolo = yolo.replace(
        "from .cmafm import CMAFM_Fusion", "from .cmafm_checkpoint import CMAFM_Fusion"
    )
    if NEW_PARSER_BLOCK in yolo:
        yolo = yolo.replace(NEW_PARSER_BLOCK, CHECKPOINT_PARSER_BLOCK)
    elif CHECKPOINT_PARSER_BLOCK not in yolo:
        raise SystemExit("Could not locate the CMAFM parser branch in models/yolo_test.py")
    parser_path.write_text(yolo, encoding="utf-8")

    print("Installed models.common.CMAFM_Fusion/_CMAFM checkpoint compatibility")


if __name__ == "__main__":
    main()

