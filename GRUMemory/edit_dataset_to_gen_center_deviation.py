#!/usr/bin/env python3
import cv2
import numpy as np
import csv
import os
import glob


ROOT = "../datasets" 


def find_all_datasets(root):

    dataset_paths = []

    # 1. train
    for d in os.listdir(root):
        full = os.path.join(root, d)
        if os.path.isdir(full) and d not in ["test", "validation"]:
            csv_path = os.path.join(full, "dataset.csv")
            if os.path.exists(csv_path):
                dataset_paths.append(csv_path)

    # 2. test
    test_dir = os.path.join(root, "test")
    if os.path.exists(test_dir):
        for d in os.listdir(test_dir):
            full = os.path.join(test_dir, d)
            csv_path = os.path.join(full, "dataset.csv")
            if os.path.exists(csv_path):
                dataset_paths.append(csv_path)

    # 3. validation
    val_dir = os.path.join(root, "validation")
    if os.path.exists(val_dir):
        for d in os.listdir(val_dir):
            full = os.path.join(val_dir, d)
            csv_path = os.path.join(full, "dataset.csv")
            if os.path.exists(csv_path):
                dataset_paths.append(csv_path)

    return dataset_paths


def process_image(img, prev_deviation):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, w, _ = rgb.shape

    mask_yellow = cv2.inRange(hsv, np.array([18, 80, 100]), np.array([40, 255, 255]))
    mask_black  = cv2.inRange(rgb, np.array([0,0,0]), np.array([50,50,50]))

    ys = np.linspace(int(h*0.2), int(h*0.95), 20).astype(int)

    points = []

    for y in ys:

        row_y = mask_yellow[y]

        in_segment = False
        start = 0

        for x in range(w):

            if row_y[x] > 0 and not in_segment:
                in_segment = True
                start = x

            elif row_y[x] == 0 and in_segment:
                end = x - 1
                in_segment = False

                left = start - 1
                right = end + 1

                if left >= 0 and right < w:
                    if mask_black[y,left] > 0 and mask_black[y,right] > 0 and (end-start)>=17:
                        cx = (start + end)//2
                        points.append((cx,y))

        if in_segment:
            end = w-1
            left = start-1
            right = end+1

            if left >= 0 and right < w:
                if mask_black[y,left] > 0 and mask_black[y,right] > 0 and (end-start)>=17:
                    cx = (start + end)//2
                    points.append((cx,y))


    if len(points) >= 4:

        pts = np.array(points)
        xs = pts[:,0]
        ys_pts = pts[:,1]

        coeffs = np.polyfit(ys_pts, xs, 2)

        y_bottom = h - 1
        x_bottom = int(coeffs[0]*y_bottom*y_bottom + coeffs[1]*y_bottom + coeffs[2])

        cx_img = w // 2
        new_deviation = cx_img - x_bottom

        if abs(new_deviation - prev_deviation) > 200:
            deviation = prev_deviation
        else:
            deviation = new_deviation

    else:
        deviation = prev_deviation

    return deviation


def process_dataset(csv_path):

    print(f"\n[INFO] Processing: {csv_path}")

    base_dir = os.path.dirname(csv_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "deviation" not in fieldnames:
        fieldnames.append("deviation")

    prev_deviation = 0

    for i, row in enumerate(rows):

        img_path = os.path.join(base_dir, row["mask_path"].lstrip("/"))

        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Unable to load: {img_path}")
            row["deviation"] = prev_deviation
            continue

        deviation = process_image(img, prev_deviation)
        prev_deviation = deviation

        row["deviation"] = deviation

        if i % 100 == 0:
            print(f"  -> {i}/{len(rows)}")

    # save
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Updated: {csv_path}")



def main():

    dataset_paths = find_all_datasets(ROOT)

    print(f"[INFO] Found {len(dataset_paths)} datasets")

    for path in dataset_paths:
        process_dataset(path)


if __name__ == "__main__":
    main()