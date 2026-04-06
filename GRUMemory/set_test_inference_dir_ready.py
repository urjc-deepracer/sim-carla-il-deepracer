import os
import csv
import shutil
import sys

NUM_IMGS = 10
DT = 0.2  # second delay in seqs
OUTPUT_DIR = "test_inference_dir"


def load_rows(folder):
    csv_path = os.path.join(folder, "dataset.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"does not exist: {csv_path}")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "timestamp" not in row:
                raise ValueError("CSV does not have 'timestamp'")
            if "mask_path" not in row:
                raise ValueError("CSV does not have 'mask_path'")

            try:
                row["_timestamp"] = float(row["timestamp"])
            except Exception:
                continue

            rows.append(row)

    if not rows:
        raise ValueError("Empty dataset or invalid timestamps")

    rows.sort(key=lambda r: r["_timestamp"])
    return rows


def build_sequence_from_start(rows, start_idx, dt, num_imgs):
    selected = []
    current_pos = start_idx
    t0 = rows[start_idx]["_timestamp"]

    for k in range(num_imgs):
        target_time = t0 + k * dt

        found_idx = None
        for j in range(current_pos, len(rows)):
            if rows[j]["_timestamp"] >= target_time:
                found_idx = j
                break

        if found_idx is None:
            return None

        selected.append(rows[found_idx])
        current_pos = found_idx + 1

    return selected


def select_sequence(folder, dt, num_imgs):
    rows = load_rows(folder)

    for start_idx in range(len(rows)):
        seq = build_sequence_from_start(rows, start_idx, dt, num_imgs)
        if seq is not None:
            return seq

    raise ValueError(
        f"Unable to build a {num_imgs} images seq with dt={dt}"
    )


def prepare_output_dir():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def copy_and_create(folder, selected_rows):
    prepare_output_dir()

    out_csv_path = os.path.join(OUTPUT_DIR, "dataset.csv")

    clean_rows = []
    for i, row in enumerate(selected_rows):
        img_rel = row["mask_path"].lstrip("/")
        img_src = os.path.join(folder, img_rel)

        if not os.path.isfile(img_src):
            raise FileNotFoundError(f"Image: {img_src} does not exist")

        new_name = f"{i:03d}.png"
        img_dst = os.path.join(OUTPUT_DIR, new_name)
        shutil.copy2(img_src, img_dst)

        new_row = dict(row)
        new_row["mask_path"] = new_name
        new_row.pop("_timestamp", None)
        clean_rows.append(new_row)

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=clean_rows[0].keys())
        writer.writeheader()
        writer.writerows(clean_rows)

    print(f"[OK] Sequence loaded on {OUTPUT_DIR}")
    print("[INFO] Selected timestamps:")
    for r in clean_rows:
        print(r["timestamp"])


def main():
    if len(sys.argv) < 2:
        print("Usage: python set_test_inference_dir_ready.py <ruta_dataset>")
        return

    folder = sys.argv[1]
    selected = select_sequence(folder, DT, NUM_IMGS)
    copy_and_create(folder, selected)


if __name__ == "__main__":
    main()