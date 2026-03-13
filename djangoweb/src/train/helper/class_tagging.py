import os
from pathlib import Path

# Nastav cestu k root priecinku datasetu
DATASET_ROOT = r"/mnt/c/Users/boris/Desktop/D_V2/WEAPON-PERSON"  # zmen na svoju cestu, napr. r"C:\datasets\person"

TAG = "m"  # zmen na "w" pre weapon, "wp" pre weapon+person

def tag_label_files(root_path: str, tag: str):
    root = Path(root_path)
    tagged = 0
    skipped = 0
    errors = 0

    # Prehladaj vsetky .txt subory rekurzivne
    for label_file in root.rglob("*.txt"):
        # Preskoc subory ktore nie su v priecinku "labels"
        if "labels" not in label_file.parts:
            continue

        try:
            content = label_file.read_text(encoding="utf-8")

            # Ak uz tag obsahuje, preskoc
            lines = content.splitlines()
            if any(line.strip() == tag for line in lines):
                skipped += 1
                continue

            # Pridaj tag na novy riadok na koniec
            if content and not content.endswith("\n"):
                content += "\n"
            content += tag + "\n"

            label_file.write_text(content, encoding="utf-8")
            tagged += 1

        except Exception as e:
            print(f"CHYBA pri {label_file}: {e}")
            errors += 1

    print(f"\nHotovo!")
    print(f"  Otagovanych:  {tagged}")
    print(f"  Preskocených: {skipped} (uz mali tag)")
    print(f"  Chyb:         {errors}")


def remove_label_files():
    LABELS_DIR = r"/mnt/c/Users/boris/Desktop/D_V2/collections/train/labels"
    TAG_TO_REMOVE = "m"  # zmen na "w", "p", "m" atd.

    removed = 0
    skipped = 0

    for label_file in Path(LABELS_DIR).glob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").splitlines()

        # Najdi posledny neprazdny riadok
        non_empty_indices = [i for i, l in enumerate(lines) if l.strip()]
        if not non_empty_indices:
            skipped += 1
            continue

        last_idx = non_empty_indices[-1]

        if lines[last_idx].strip() == TAG_TO_REMOVE:
            lines.pop(last_idx)
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            removed += 1
        else:
            skipped += 1

    print(f"Odstránený tag '{TAG_TO_REMOVE}' z: {removed} súborov")
    print(f"Preskočených (tag nenájdený): {skipped}")






if __name__ == "__main__":
    tag_label_files(DATASET_ROOT, TAG)