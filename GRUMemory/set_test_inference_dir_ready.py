import os
import csv
import shutil

NUM_IMGS = 10
DT = 0.2  # segundos
OUTPUT_DIR = "test_inference_dir"


def select_sequence(folder, dt, num_imgs):

    csv_path = os.path.join(folder, "dataset.csv")

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("Dataset vacío")

    selected = []
    start_idx = 0

    while start_idx < len(rows):

        selected = [rows[start_idx]]
        last_time = float(rows[start_idx]["timestamp"])

        for i in range(start_idx + 1, len(rows)):
            t = float(rows[i]["timestamp"])

            if t - last_time >= dt:
                selected.append(rows[i])
                last_time = t

            if len(selected) == num_imgs:
                return selected

        start_idx += 1

    raise ValueError("No se pudo construir secuencia con ese dt")


def copy_and_create(folder, selected_rows):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_csv_path = os.path.join(OUTPUT_DIR, "dataset.csv")

    with open(out_csv_path, "w", newline="") as f:
        writer = None

        for i, row in enumerate(selected_rows):

            img_rel = row["mask_path"].lstrip("/")
            img_src = os.path.join(folder, img_rel)

            new_name = f"{i:03d}.png"
            img_dst = os.path.join(OUTPUT_DIR, new_name)

            shutil.copy(img_src, img_dst)

            row["mask_path"] = new_name

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()

            writer.writerow(row)

    print(f"[OK] Secuencia guardada en {OUTPUT_DIR}")


def main():

    import sys

    if len(sys.argv) < 2:
        print("Uso: python set_test_inference_dir_ready.py <ruta_dataset>")
        return

    folder = sys.argv[1]

    selected = select_sequence(folder, DT, NUM_IMGS)

    copy_and_create(folder, selected)


if __name__ == "__main__":
    main()