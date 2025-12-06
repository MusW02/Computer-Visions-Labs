import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import time
import os
import random
from PIL import ImageFile

# --- FIX FOR CORRUPT IMAGES ---
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURATION ---
DATA_DIR = "data/PetImages"  # Make sure this matches your folder structure
BATCH_SIZE = 32
NUM_EPOCHS = 3               # Reduced to 3 epochs for speed
LEARNING_RATE = 1e-4
NUM_CLASSES = 2 
SUBSET_SIZE = 1000           # Only use 1000 images total (Drastic speedup)

# --- 1. SETUP DEVICE & TRANSFORMS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. LOAD & SUBSET DATASET ---
def get_data_loaders(data_dir):
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found.")
        return None, None

    # Load the entire dataset structure
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # --- SUBSETTING LOGIC (THE FIX) ---
    total_images = len(full_dataset)
    indices = list(range(total_images))
    random.shuffle(indices)
    
    # Take only the first SUBSET_SIZE images (e.g., 1000)
    subset_indices = indices[:SUBSET_SIZE]
    small_dataset = Subset(full_dataset, subset_indices)
    
    print(f" Original Dataset Size: {total_images}")
    print(f" Reduced Dataset Size: {len(small_dataset)} (for faster CPU training)")
    # ----------------------------------

    # Split the small dataset into Train (80%) and Val (20%)
    train_size = int(0.8 * len(small_dataset))
    val_size = len(small_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(small_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" Classes: {full_dataset.classes}")
    return train_loader, val_loader

# --- 3. INITIALIZE ViT MODEL ---
def get_vit_model(num_classes):
    print("Loading ViT-Base-16...")
    # Using a slightly lighter model if available, or standard base
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model.to(device)

# --- 4. TRAINING LOOP ---
def train_model():
    train_loader, val_loader = get_data_loaders(DATA_DIR)
    if not train_loader: return

    model = get_vit_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            batch_count += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 5 batches
            if batch_count % 5 == 0:
                print(f" Epoch {epoch+1}: Processed batch {batch_count}/{len(train_loader)}...")
            
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

    print("\nTraining Complete.")

if __name__ == "__main__":
    train_model()