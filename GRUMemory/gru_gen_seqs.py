import os, glob, csv, torch
from PIL import Image
from torchvision import transforms


ROOT = "../datasets"
SEQ_LEN = 10
DT = 0.2

TRAIN_PATH = "train.pt"
VAL_PATH   = "val.pt"
TEST_PATH  = "test.pt"

transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor()
])


def generate_and_save(base_path, save_path):

    folders = sorted(glob.glob(os.path.join(base_path, "Deepracer_*")))

    all_seq_imgs = []
    all_seq_labels = []
    all_seq_estados = []
    all_seq_speeds = []
    all_seq_deviations = []
    all_seq_controls = []

    for folder in folders:

        print("[INFO] Processing:", folder)

        csv_path = os.path.join(folder, "dataset.csv")
        if not os.path.exists(csv_path):
            continue

        images, labels, estados, speeds, deviations, controls = [], [], [], [], [], []

        with open(csv_path) as f:
            reader = csv.DictReader(f)

            for row in reader:
                img_path = os.path.join(folder, row["mask_path"].lstrip("/"))

                if not os.path.isfile(img_path):
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img)
                except:
                    continue

                steer = float(row["steer"])
                throttle = float(row["throttle"])
                estado = float(row["estado"])
                speed = float(row["speed"])
                deviation = float(row["deviation"])

                images.append(img)
                labels.append([steer, throttle])
                estados.append(estado)
                speeds.append(speed)
                deviations.append(deviation)
                controls.append([steer, throttle])


        if len(images) == 0:
            continue

        images = torch.stack(images)
        labels = torch.tensor(labels)
        estados = torch.tensor(estados)
        speeds = torch.tensor(speeds)
        deviations = torch.tensor(deviations)
        controls = torch.tensor(controls)

        i = 0
        # 
        while i + (SEQ_LEN - 1) * DT < len(images):
            idxs = [int(i + k * DT) for k in range(SEQ_LEN)]
            seq = images[idxs]
            seq_speeds = speeds[idxs[:-1]]
            seq_deviations = deviations[idxs]
            seq_controls = controls[idxs[:-1]]   # hasta t-1

            end = idxs[-1]

            all_seq_imgs.append(seq)
            all_seq_labels.append(labels[end])
            all_seq_estados.append(estados[end])
            all_seq_speeds.append(seq_speeds)
            all_seq_deviations.append(seq_deviations)
            all_seq_controls.append(seq_controls)

            i += SEQ_LEN * DT

    torch.save({
        "images": all_seq_imgs,
        "speeds": all_seq_speeds,
        "controls": all_seq_controls,
        "labels": torch.stack(all_seq_labels),
        "estados": torch.stack(all_seq_estados),
        "deviations": torch.stack(all_seq_deviations)
    }, save_path)

    print(f"[OK] Saved {save_path} with {len(all_seq_imgs)} sequencies\n")


def load_sequences(path):
    data = torch.load(path)
    return (
        data["images"],
        data["speeds"],
        data["deviations"],
        data["controls"], 
        data["labels"],
        data["estados"]
    )

if not os.path.exists(TRAIN_PATH):
    generate_and_save(ROOT, TRAIN_PATH)

if not os.path.exists(VAL_PATH):
    generate_and_save(os.path.join(ROOT, "validation"), VAL_PATH)

if not os.path.exists(TEST_PATH):
    generate_and_save(os.path.join(ROOT, "test"), TEST_PATH)

train_imgs, train_speeds, train_labels, train_estados = load_sequences(TRAIN_PATH)
val_imgs, val_speeds, val_labels, val_estados = load_sequences(VAL_PATH)
test_imgs, test_speeds, test_labels, test_estados = load_sequences(TEST_PATH)

print("====================================")
print(f"Train: {len(train_imgs)} | speeds: {len(train_speeds)}")
print(f"Val:   {len(val_imgs)} | speeds: {len(val_speeds)}")
print(f"Test:  {len(test_imgs)} | speeds: {len(test_speeds)}")
print("====================================")