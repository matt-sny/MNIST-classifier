import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split

@dataclass
class MyDataset(TorchDataset):
    image_paths: Sequence[str]
    labels: Sequence[int]
    transform: Optional[Callable] = None

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(('.jpeg', '.jpg', '.png'))

def load_image_paths_and_labels(
    data_path: str,
    label_categories: dict,
) -> Tuple[List[str], List[int]]:

    image_paths: List[str] = []
    labels: List[int] = []

    for label_folder, label_idx in label_categories.items():
        folder_path = os.path.join(data_path, label_folder)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Expected directory for label '{label_folder}' not found: {folder_path}")

        entries = sorted(os.listdir(folder_path))
        image_files = [f for f in entries if is_image_file(f)]

        full_paths = [os.path.join(folder_path, fname) for fname in image_files]
        image_paths.extend(full_paths)
        labels.extend([label_idx] * len(full_paths))

    if len(image_paths) != len(labels):
        raise ValueError("Mismatch between number of images and labels")

    return image_paths, labels

def create_dataloaders(
    data_path: str,
    label_categories: dict,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    batch_size: int = 1,
    split_sizes: Optional[Tuple[int, int, int]] = None
):
    start_time = time.time()

    print("Creating DataLoaders...")

    def _resolve_split_dir(root: str, candidates: Sequence[str]) -> str:
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(f"Expected one of split directories {candidates} under {root}, but none were found.")

    datasets = {}
    for i, (logical_name, candidates, transform) in enumerate((
        ("train", ("train",), train_transform),
        ("val", ("val", "validation"), val_transform),
        ("test", ("test",), test_transform)),
    ):
        split_dir = _resolve_split_dir(data_path, candidates)
        image_paths, labels = load_image_paths_and_labels(data_path=split_dir, label_categories=label_categories)
        if split_sizes is not None:
            size = split_sizes[i]
            if size > len(image_paths):
                raise ValueError(f"Requested size {size} exceeds available images ({len(image_paths)}) in {logical_name} split.")
            elif size < 1:
                raise ValueError(f"Requested size {size} is less than 1 in {logical_name} split.")
            image_paths_sub, _, labels_sub, _ = train_test_split(image_paths, labels, train_size=size, stratify=labels, random_state=42)
            image_paths = image_paths_sub
            labels = labels_sub
        datasets[logical_name] = MyDataset(image_paths, labels, transform=transform)

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    elapsed = time.time() - start_time
    print(f"Created DataLoaders in {elapsed:.1f} seconds")

    return train_loader, val_loader, test_loader
