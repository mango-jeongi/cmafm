"""Create a CMAFM evaluation manifest from the aligned FLIR official test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = ("People", "Car", "Bus", "Motorcycle", "Lamp", "Truck")
FLIR_CLASS_MAP = {0: 1, 1: 0}  # FLIR car -> CMAFM Car; FLIR person -> CMAFM People


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = args.root / "coco_annotations" / "test.json"
    visible_dir = args.root / "visible" / "test"
    thermal_dir = args.root / "thermal" / "test"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))

    annotations_by_image: defaultdict[object, list] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] in FLIR_CLASS_MAP:
            annotations_by_image[annotation["image_id"]].append(annotation)

    records = []
    class_counts: Counter[str] = Counter()
    skipped_without_mapped_labels = 0
    for image in sorted(coco["images"], key=lambda item: str(item["id"])):
        mapped_annotations = annotations_by_image[image["id"]]
        if not mapped_annotations:
            skipped_without_mapped_labels += 1
            continue

        filename = image["file_name"]
        rgb_path = visible_dir / filename
        thermal_path = thermal_dir / filename
        if not rgb_path.is_file() or not thermal_path.is_file():
            raise FileNotFoundError(f"Missing aligned pair for {filename}")

        width = int(image["width"])
        height = int(image["height"])
        annotations = []
        for annotation in mapped_annotations:
            class_id = FLIR_CLASS_MAP[int(annotation["category_id"])]
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1 = min(max(x, 0.0), width)
            y1 = min(max(y, 0.0), height)
            x2 = min(max(x + box_width, 0.0), width)
            y2 = min(max(y + box_height, 0.0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            annotations.append({"class_id": class_id, "bbox_xyxy": [x1, y1, x2, y2]})
            class_counts[CLASS_NAMES[class_id]] += 1

        if not annotations:
            skipped_without_mapped_labels += 1
            continue
        records.append({
            "id": str(image["id"]),
            "rgb": str(rgb_path.relative_to(args.output.parent)).replace("\\", "/"),
            "thermal": str(thermal_path.relative_to(args.output.parent)).replace("\\", "/"),
            "width": width,
            "height": height,
            "annotations": annotations,
        })

    manifest = {
        "dataset": "FLIR Aligned",
        "split": "official test",
        "seed": None,
        "class_mapping": {
            "FLIR person (1)": "People (0)",
            "FLIR car (0)": "Car (1)",
        },
        "ignored_source_classes": ["bicycle", "dog"],
        "source_image_count": len(coco["images"]),
        "validation_image_count": len(records),
        "skipped_without_mapped_labels": skipped_without_mapped_labels,
        "class_names": list(CLASS_NAMES),
        "class_box_counts": dict(class_counts),
        "records": records,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {args.output}")
    print(f"Evaluation images: {len(records)}")
    print(f"Skipped images without mapped labels: {skipped_without_mapped_labels}")
    print(f"Ground-truth boxes: {sum(class_counts.values())}")
    print(f"Class counts: {dict(class_counts)}")


if __name__ == "__main__":
    main()
