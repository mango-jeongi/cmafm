"""Create the deterministic M3FD validation subset without extracting the full archive."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


CLASS_NAMES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def png_size(stream) -> tuple[int, int]:
    header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected a PNG image")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between zero and one")

    rgb_dir = args.output / "rgb"
    thermal_dir = args.output / "thermal"
    rgb_dir.mkdir(parents=True)
    thermal_dir.mkdir(parents=True)

    records: list[dict] = []
    class_counts: Counter[str] = Counter()

    with zipfile.ZipFile(args.archive) as archive:
        annotation_names = sorted(
            name for name in archive.namelist()
            if name.startswith("Annotation/") and name.lower().endswith(".xml")
        )
        stems = [Path(name).stem for name in annotation_names]
        random.Random(args.seed).shuffle(stems)
        validation_stems = sorted(stems[: int(len(stems) * args.val_ratio)])

        for index, stem in enumerate(validation_stems, start=1):
            xml_name = f"Annotation/{stem}.xml"
            rgb_name = f"Vis/{stem}.png"
            thermal_name = f"Ir/{stem}.png"
            for name in (xml_name, rgb_name, thermal_name):
                archive.getinfo(name)

            root = ET.fromstring(archive.read(xml_name))
            xml_width = int(root.findtext("size/width", "0"))
            xml_height = int(root.findtext("size/height", "0"))
            with archive.open(rgb_name) as stream:
                rgb_width, rgb_height = png_size(stream)
            with archive.open(thermal_name) as stream:
                thermal_width, thermal_height = png_size(stream)
            if (rgb_width, rgb_height) != (thermal_width, thermal_height):
                raise ValueError(f"Unaligned pair {stem}: RGB and thermal sizes differ")
            if (xml_width, xml_height) != (rgb_width, rgb_height):
                raise ValueError(f"Annotation size mismatch for {stem}")

            annotations = []
            for obj in root.findall("object"):
                class_name = (obj.findtext("name") or "").strip()
                if class_name not in CLASS_IDS:
                    continue
                box = obj.find("bndbox")
                if box is None:
                    continue
                xyxy = [
                    float(box.findtext("xmin", "0")),
                    float(box.findtext("ymin", "0")),
                    float(box.findtext("xmax", "0")),
                    float(box.findtext("ymax", "0")),
                ]
                if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                    raise ValueError(f"Invalid box in {xml_name}: {xyxy}")
                annotations.append({"class_id": CLASS_IDS[class_name], "bbox_xyxy": xyxy})
                class_counts[class_name] += 1

            rgb_target = rgb_dir / f"{stem}.png"
            thermal_target = thermal_dir / f"{stem}.png"
            with archive.open(rgb_name) as source, rgb_target.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            with archive.open(thermal_name) as source, thermal_target.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

            records.append({
                "id": stem,
                "rgb": f"rgb/{stem}.png",
                "thermal": f"thermal/{stem}.png",
                "width": rgb_width,
                "height": rgb_height,
                "annotations": annotations,
            })
            if index % 100 == 0:
                print(f"Prepared {index}/{len(validation_stems)} pairs", flush=True)

    manifest = {
        "dataset": "M3FD",
        "source_archive": args.archive.name,
        "split_method": "sorted stems, Python random shuffle, first floor(N * ratio)",
        "seed": args.seed,
        "validation_ratio": args.val_ratio,
        "source_image_count": len(annotation_names),
        "validation_image_count": len(records),
        "class_names": list(CLASS_NAMES),
        "class_box_counts": dict(class_counts),
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Validation images: {len(records)}")
    print(f"Ground-truth boxes: {sum(class_counts.values())}")
    print(f"Class counts: {dict(class_counts)}")


if __name__ == "__main__":
    main()
