#!/usr/bin/env python3
import cv2
import numpy as np
import csv
import os
import sys

prev_deviation = 0

def process_image(img,prev_deviation):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, w, _ = rgb.shape

    lower_yellow = np.array([18, 80, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_black = cv2.inRange(rgb, np.array([0, 0, 0]), np.array([50, 50, 50]))

    ys = np.linspace(int(h*0.2), int(h*0.95), 20).astype(int)

    points = []

    for y in ys:

        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)

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
                    if mask_black[y, left] > 0 and mask_black[y, right] > 0 and (end - start) >= 17:

                        cx = (start + end) // 2
                        points.append((cx, y))
                        cv2.circle(img, (cx, y), 6, (0, 0, 255), -1)

        if in_segment:
            end = w - 1
            left = start - 1
            right = end + 1

            if left >= 0 and right < w:
                if mask_black[y, left] > 0 and mask_black[y, right] > 0 and (end - start) >= 17:

                    cx = (start + end) // 2
                    points.append((cx, y))
                    cv2.circle(img, (cx, y), 6, (0, 0, 255), -1)

    deviation_px = 0
    x_bottom = None

    if len(points) >= 4:

        pts = np.array(points)
        xs = pts[:, 0]
        ys_pts = pts[:, 1]

        coeffs = np.polyfit(ys_pts, xs, 2)

        for y in range(int(min(ys_pts)), int(max(ys_pts))):
            x = int(coeffs[0]*y*y + coeffs[1]*y + coeffs[2])
            if 0 <= x < w:
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

        y_bottom = h - 1
        x_bottom = int(coeffs[0]*y_bottom*y_bottom + coeffs[1]*y_bottom + coeffs[2])

        cv2.circle(img, (x_bottom, y_bottom), 10, (255, 255, 0), -1)

        cx_img = w // 2
        cv2.line(img, (cx_img, 0), (cx_img, h), (255, 0, 0), 2)

        deviation_px = cx_img - x_bottom

        cv2.line(img, (cx_img, y_bottom), (x_bottom, y_bottom), (0, 255, 255), 2)

        status = "OK"

    else:
        deviation_px = prev_deviation
        status = "USING PREV"

    cv2.putText(img, f"Deviation: {deviation_px}px",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 255), 2)

    cv2.putText(img, f"Points: {len(points)} | {status}",
            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0) if status=="OK" else (0,0,255), 2)

    return img, deviation_px


def visualize_dataset(csv_path):

    global prev_deviation
    
    base_dir = os.path.dirname(csv_path)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] Total frames: {len(rows)}")

    i = 0

    while True:

        row = rows[i]
        img_path = os.path.join(base_dir, row["mask_path"].lstrip("/"))

        img = cv2.imread(img_path)

        if img is None:
            print(f"[ERROR] Unable to load: {img_path}")
            i += 1
            continue

        vis, deviation = process_image(img, prev_deviation)
        prev_deviation = deviation

        cv2.imshow("Dataset Visualizer", vis)

        print(f"[Frame {i}] Deviation: {deviation}")

        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('a'):
            i = max(0, i - 1)
        else:
            i = min(len(rows) - 1, i + 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 deviation_visualizer.py dataset.csv")
        exit()

    visualize_dataset(sys.argv[1])