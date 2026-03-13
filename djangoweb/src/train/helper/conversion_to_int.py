from pathlib import Path

LABELS_DIR = r"/mnt/c/Users/boris/Desktop/D_V2/collections/train/labels_aug"

fixed = 0
skipped = 0

for label_file in Path(LABELS_DIR).glob("*.txt"):
    lines = label_file.read_text(encoding="utf-8").splitlines()
    new_lines = []
    changed = False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            new_lines.append(line)
            continue

        # Ak je prvy element float (napr. "0.0", "1.0"), zmen na int
        try:
            first = parts[0]
            if '.' in first:
                parts[0] = str(int(float(first)))
                changed = True
        except ValueError:
            pass  # tag ako "w", "p" - nechaj tak

        new_lines.append(" ".join(parts))

    if changed:
        label_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        fixed += 1
    else:
        skipped += 1

print(f"Opravených: {fixed}")
print(f"Preskočených (už bolo OK): {skipped}")