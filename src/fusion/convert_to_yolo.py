import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths on HPC-Cluster
# Note: Update this path if the data is located elsewhere on the cluster
root = Path("src/fusion/data/M3FD")
ann_dir = root / "Annotation"
out_dir = root / "labels"
os.makedirs(out_dir, exist_ok=True)

classes = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]

def convert(size, box):
    """Converts VOC bounding box to YOLO format (normalized xywh)."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def main():
    print(f"Starting conversion of VOC XMLs in: {ann_dir}")
    count = 0
    for xml_file in ann_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root_node = tree.getroot()
            size = root_node.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            with open(out_dir / (xml_file.stem + ".txt"), "w") as f:
                for obj in root_node.findall('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert((w, h), b)
                    f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")
            count += 1
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print(f"Conversion complete. {count} YOLO labels saved to {out_dir}")

if __name__ == "__main__":
    main()
