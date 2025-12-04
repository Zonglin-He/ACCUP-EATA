import torch

for split in ["train_0", "train_1", "train_2", "train_3", "test_0", "test_1", "test_2", "test_3"]:
    obj = torch.load(f"D:/PyCharm Project/ACCUP + EATA/data/Dataset/FD/{split}.pt", map_location="cpu")
    print(split, type(obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "shape"):
                print("  ", k, v.shape, v.dtype if hasattr(v, "dtype") else "")
            elif isinstance(v, list):
                print("  ", k, "list len", len(v))
            else:
                print("  ", k, type(v))
    print("---")