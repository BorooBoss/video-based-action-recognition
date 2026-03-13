from pathlib import Path
from collections import defaultdict

DATASET_ROOT = r"/mnt/c/Users/boris/Desktop/D_V2/collections"

CLASS_NAMES = {
    0: "person",
    1: "weapon",
}

def count_annotations(root_path: str):
    root = Path(root_path)
    stats = defaultdict(int)
    tag_counts = defaultdict(int)
    empty_files = 0

    labels_dir = root / "train" / "labels"
    if not labels_dir.exists():
        print(f"Nenasiel som: {labels_dir}")
        return

    label_files = list(labels_dir.glob("*.txt"))

    for label_file in label_files:
        content = label_file.read_text(encoding="utf-8").strip()
        lines = content.splitlines()

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts[0].lstrip('-').isdigit():
                tag_counts[parts[0]] += 1
                continue
            try:
                stats[int(parts[0])] += 1
            except ValueError:
                continue

    print("=" * 50)
    print(f"TRAIN SET — {len(label_files)} suborov")
    print("=" * 50)
    total = 0
    for class_id in sorted(stats.keys()):
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        print(f"  class {class_id} ({name}): {stats[class_id]} anotacii")
        total += stats[class_id]
    print(f"  SPOLU: {total} anotacii")
    if tag_counts:
        print(f"  Tagy: {dict(tag_counts)}")
    if empty_files:
        print(f"  Prazdnych suborov: {empty_files}")
    print("=" * 50)

if __name__ == "__main__":
    count_annotations(DATASET_ROOT)