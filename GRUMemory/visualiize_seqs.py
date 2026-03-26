#!/usr/bin/env python3
import os, glob, csv
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


ROOT = "../datasets"
SEQ_LEN = 10
DT_DATASET = 1/30

COMMON_TF = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def denormalize(img):
    img = img.clone()
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)


def load_dataset(folder):

    csv_path = os.path.join(folder, "dataset.csv")
    if not os.path.exists(csv_path):
        return None

    images, labels = [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            img_path = os.path.join(folder, row["rgb_path"].lstrip("/"))

            if not os.path.isfile(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            img = COMMON_TF(img)

            steer = float(row["steer"])
            throttle = float(row["throttle"])

            images.append(img)
            labels.append([steer, throttle])

    if len(images) == 0:
        return None

    return torch.stack(images), torch.tensor(labels)


class Visualizer:

    def __init__(self, root):

        self.folders = sorted(glob.glob(os.path.join(root, "Deepracer_*")))
        self.dataset_idx = 0

        self.seconds_step = 0.1
        self.frame_jump = 1

        self.load_dataset()

        self.seq_idx = 0

        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_jump()
        self.update()

    def update_jump(self):
        self.frame_jump = max(1, int(self.seconds_step / DT_DATASET))

    def load_dataset(self):
        self.current_folder = self.folders[self.dataset_idx]
        print(f"\n[DATASET] {self.current_folder}")

        data = load_dataset(self.current_folder)

        if data is None:
            self.images, self.labels = [], []
        else:
            self.images, self.labels = data

        self.seq_idx = 0

    def get_sequence(self, start_idx):

        seq_imgs = []
        label = None

        for k in range(SEQ_LEN):
            idx = start_idx + k * self.frame_jump

            if idx >= len(self.images):
                break

            seq_imgs.append(self.images[idx])
            label = self.labels[idx]

        if len(seq_imgs) == 0:
            return None

        return torch.stack(seq_imgs), label

    def update(self):

        if len(self.images) == 0:
            return

        plt.clf()

        seq_data = self.get_sequence(self.seq_idx)

        if seq_data is None:
            return

        seq, label = seq_data

        T = seq.shape[0]

        for t in range(T):
            img = denormalize(seq[t])
            img = img.permute(1,2,0).numpy()

            plt.subplot(1, T, t+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{t}")

        steer = label[0].item()
        throttle = label[1].item()

        plt.suptitle(
            f"{os.path.basename(self.current_folder)} | "
            f"Seq {self.seq_idx}/{len(self.images)} | Δt={self.seconds_step:.3f}s\n"
            f"steer={steer:.3f} throttle={throttle:.3f}",
            fontsize=12
        )

        plt.draw()

    def on_key(self, event):

        # navegación fluida
        if event.key == "right":
            self.seq_idx = min(self.seq_idx + 1, len(self.images)-1)

        elif event.key == "left":
            self.seq_idx = max(self.seq_idx - 1, 0)

        # salto rápido
        elif event.key == "pageup":
            self.seq_idx = min(self.seq_idx + 50, len(self.images)-1)

        elif event.key == "pagedown":
            self.seq_idx = max(self.seq_idx - 50, 0)

        # cambiar dataset
        elif event.key == "up":
            self.dataset_idx = (self.dataset_idx + 1) % len(self.folders)
            self.load_dataset()

        elif event.key == "down":
            self.dataset_idx = (self.dataset_idx - 1) % len(self.folders)
            self.load_dataset()

        # cambiar tiempo
        elif event.key == "w":
            self.seconds_step += 0.05
            self.update_jump()

        elif event.key == "s":
            self.seconds_step = max(0.01, self.seconds_step - 0.05)
            self.update_jump()

        # 🔥 SALTO DIRECTO POR INPUT
        elif event.key == "enter":
            try:
                idx = int(input("Ir a secuencia: "))
                self.seq_idx = max(0, min(idx, len(self.images)-1))
            except:
                print("Input inválido")

        elif event.key == "q":
            plt.close(self.fig)

        self.update()


# ==========================================
# MAIN
# ==========================================
def main():

    vis = Visualizer(ROOT)

    print("""
🎮 CONTROLES:
→ / ← navegar
PageUp / PageDown saltos grandes
↑ ↓ cambiar dataset
w / s cambiar Δt
ENTER escribir índice
q salir
""")

    plt.show()


if __name__ == "__main__":
    main()