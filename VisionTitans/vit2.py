import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Patch Embedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# --- Persistent Memory ---
class PersistentMemoryEmbedding(nn.Module):
    def __init__(self, memory_size, embed_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))

    def forward(self, x):
        batch_size = x.size(0)
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([memory, x], dim=1)

# --- Gated Memory Transition ---
class GatedMemoryTransition(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, mem):
        combined = torch.cat([x, mem], dim=-1)
        g = self.gate(combined)
        return g * x + (1 - g) * mem

# --- Residual Norm ---
class ResidualNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """Apply function with residual connection.

        Extra positional and keyword arguments are forwarded to ``self.fn`` so
        that wrapped functions can accept additional inputs (e.g. memory
        tensors).
        """
        return x + self.fn(self.norm(x), *args, **kwargs)

# --- Transformer Encoder ---
class MultiScaleEncoder(nn.Module):
    def __init__(self, emb_dim=64, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.transpose(0, 1)
        out = self.encoder(x)
        return out.transpose(0, 1)

# --- Cross-Scale Attention ---
class CrossScaleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

    def forward(self, x1, x2):
        out, _ = self.attn(x1, x2, x2)
        return out

# --- TITAN-style Image Classifier ---
class TITANImageClassifier(nn.Module):
    def __init__(self, in_channels=1, patch_sizes=[4, 7], emb_dim=64, num_classes=10, memory_size=4):
        super().__init__()
        self.embedders = nn.ModuleList([
            PatchEmbedding(in_channels, patch_size=p, emb_dim=emb_dim) for p in patch_sizes
        ])
        self.memories = nn.ModuleList([
            PersistentMemoryEmbedding(memory_size, emb_dim) for _ in patch_sizes
        ])
        self.encoders = nn.ModuleList([
            MultiScaleEncoder(emb_dim=emb_dim) for _ in patch_sizes
        ])
        self.memory_transitions = nn.ModuleList([
            ResidualNorm(emb_dim, GatedMemoryTransition(emb_dim)) for _ in patch_sizes
        ])
        self.cross_attn = CrossScaleAttention(emb_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        encoded_scales = []
        for embedder, memory, encoder, memory_transition in zip(
            self.embedders, self.memories, self.encoders, self.memory_transitions
        ):
            patches = embedder(x)
            patches_with_mem = memory(patches)
            encoded = encoder(patches_with_mem)
            mem_applied = memory_transition(encoded, patches_with_mem)
            encoded_scales.append(mem_applied)

        x_combined = self.cross_attn(encoded_scales[0], encoded_scales[1])
        x_pooled = x_combined.mean(dim=1)
        return self.classifier(x_pooled)



# --- Simple ViT for comparison ---
class SimpleViT(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_dim=64, num_classes=10):
        super().__init__()
        self.embedding = PatchEmbedding(in_channels, patch_size, emb_dim)
        self.encoder = MultiScaleEncoder(emb_dim=emb_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        patches = self.embedding(x)
        encoded = self.encoder(patches)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

# --- Training Function ---
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --- Main ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    models = {
        "ViT": SimpleViT().to(device),
        "TITANs": TITANImageClassifier().to(device)
    }

    results = {}
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        print(f"Training {name}...")
        for epoch in range(5):
            train_model(model, train_loader, optimizer, criterion, device)

        acc = evaluate_model(model, test_loader, device)
        results[name] = acc
        print(f"{name} Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
