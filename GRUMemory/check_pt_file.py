#!/usr/bin/env python3
import torch


PT_PATH = "train.pt"   # file to read


def inspect_pt(path):

    print(f"\n[INFO] Loading: {path}")
    data = torch.load(path, map_location="cpu")

    print("\n================ Content ================\n")

    print("Type:", type(data))


    if isinstance(data, dict):

        print("\Keys:")
        for k in data.keys():
            print(f" - {k}")

        print("\n========== DETAILs ==========\n")

        for k, v in data.items():

            print(f"\n🔹 {k}")

            # Tensor
            if isinstance(v, torch.Tensor):
                print("  Type: Tensor")
                print("  Shape:", v.shape)
                print("  Dtype:", v.dtype)
                print("  Example:", v[0] if v.numel() > 0 else "empty")

            # List
            elif isinstance(v, list):
                print("  Type: List")
                print("  Length:", len(v))

                if len(v) > 0:
                    first = v[0]

                    if isinstance(first, torch.Tensor):
                        print("  Shape first element:", first.shape)
                    else:
                        print("  Tipo first element:", type(first))

            else:
                print("  Type:", type(v))

    else:
        print("Content:", data)

    print("\n===========================================\n")


if __name__ == "__main__":
    inspect_pt(PT_PATH)