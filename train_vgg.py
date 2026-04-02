import os
import time
import random
from typing import List, Tuple
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ai_dataset = [r"D:\DATASET\AI\train", r"D:\DATASET\AI\train_2",]
real_dataset = [r"D:\DATASET\real", r"D:\DATASET\real_2\train\real", r"D:\DATASET\real_2\train_2\real", r"D:\DATASET\real_4",]

sz = 224, btch = 16, epoch = 10, alpha = 1e-4, num_wrk = 4, model_path = "trained_resnet18_binary.pth"
whitelist = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
ai = 1, real = 0, maxi = 60_000_000

def images(folder: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(whitelist):
                files.append(os.path.join(root, name))
    return files

def check(path: str) -> bool:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return (w * h) <= maxi
    except Exception:
        return False

def img_open(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")

class BinaryImageDataset(Dataset):
    def __init__(self, ai_dataset: List[str], real_dataset: List[str], transform = None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        ai_imgs = []
        for folder in ai_dataset:
            ai_imgs.extend(images(folder))
        real_imgs = []
        for folder in real_dataset:
            real_imgs.extend(images(folder))

        raw = [(path, ai) for path in ai_imgs] + [(path, real) for path in real_imgs]
        print("\nChecking...")

        skips = 0
        for path, label in tqdm(raw, desc = "Filtering"):
            if check(path):
                self.samples.append((path, label))
            else:
                skips += 1

        if len(self.samples) == 0:
            raise ValueError("No valid samples found")

        random.shuffle(self.samples)
        cnt_ai = sum(1 for _, label in self.samples if label == ai)
        cnt_real = sum(1 for _, label in self.samples if label == real)
        print("\Results ->")
        print(f"Ai samples : {cnt_ai}")
        print(f"Real samples : {cnt_real}")
        print(f"Total : {len(self.samples)}")
        print(f"Skipped bad files : {skips}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = img_open(path)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor([label], dtype = torch.float32)
        return image, label

def one_train(model, loader, criterion, optimizer, device, epoch, total_epoch):
    model.train()

    temp = 0.0, ans = 0, total = 0
    progress = tqdm(loader, desc = f"Epoch {epoch} / {total_epoch} [Train]", leave = True)

    for images, labels in progress:
        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        temp += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        ans += (preds == labels).sum().item()
        total += labels.size(0)
        current_loss = temp / total
        current_acc = 100.0 * ans / total
        progress.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})
    epoch_loss = temp / total
    epoch_accuracy = 100.0 * ans / total
    return epoch_loss, epoch_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale = (0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    dataset = BinaryImageDataset(ai_dataset = ai_dataset, real_dataset = real_dataset, transform = train_transform)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(dataset, btch = btch, shuffle = True, num_wrk = num_wrk, pin = pin, persistent_workers = True if num_wrk > 0 else False)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights = weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), alpha = alpha)
    start_time = time.time()

    for epoch in range(1, epoch + 1):
        print(f"\nEpoch [{epoch} / {epoch}]")
        loss, accuracy = one_train(model, train_loader, criterion, optimizer, device, epoch, epoch)
        print(f"Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), model_path)
    print(f"\n Final trained model saved to {model_path}")
    print(f"Training complete in { (time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()