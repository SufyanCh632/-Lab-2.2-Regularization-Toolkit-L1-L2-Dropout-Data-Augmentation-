# -Lab-2.2-Regularization-Toolkit-L1-L2-Dropout-Data-Augmentation-

#1. Setup a compact CNN
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_tfm = transforms.ToTensor()
train_full = datasets.FashionMNIST('data', train=True, download=True, transform=base_tfm)
test = datasets.FashionMNIST('data', train=False, download=True, transform=base_tfm)
n_val = 5000
train, val = random_split(train_full, [len(train_full)-n_val, n_val], generator=torch.Generator().manual_seed(42))
def loaders(dataset, bs=128):
    return (DataLoader(dataset, batch_size=bs, shuffle=True),
            DataLoader(val, batch_size=bs),
            DataLoader(test, batch_size=bs))
train_dl, val_dl, test_dl = loaders(train)
class SmallCNN(nn.Module):
    def __init__(self, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 10)
        )
    def forward(self,x): return self.net(x)

#2. Training & evaluation utilities
def train_epochs(model, opt, train_dl, val_dl, epochs=5, l1_lambda=0.0):
    model.to(device); hist=[]
    for ep in range(epochs):
        model.train(); tot=0
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if l1_lambda>0:
                l1 = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda*l1
            loss.backward(); opt.step()
            tot += loss.item()*x.size(0)
        tr_loss = tot/len(train_dl.dataset)
        vl_loss, vl_acc = evaluate(model, val_dl)
        hist.append((tr_loss, vl_loss, vl_acc))
        print(f"ep{ep+1} train {tr_loss:.3f}  val {vl_loss:.3f}  acc {vl_acc:.3f}")
    return hist
@torch.no_grad()
def evaluate(model, dl):
    model.eval(); tot=0; correct=0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        logits = model(x); loss = F.cross_entropy(logits, y)
        tot += loss.item()*x.size(0)
        correct += (logits.argmax(1)==y).sum().item()
    return tot/len(dl.dataset), correct/len(dl.dataset)

#3. Baseline (no regularization)
m0 = SmallCNN(p_drop=0.0).to(device)
opt0 = optim.Adam(m0.parameters(), lr=1e-3)
print("Baseline:")
_ = train_epochs(m0, opt0, train_dl, val_dl, epochs=5)

#4. L2(weight decay)
m_l2 = SmallCNN(p_drop=0.0).to(device)
opt_l2 = optim.Adam(m_l2.parameters(), lr=1e-3, weight_decay=1e-4)
print("L2 weight decay:")
_ = train_epochs(m_l2, opt_l2, train_dl, val_dl, epochs=5)

#5. L1 penalty(manual)
m_l1 = SmallCNN(p_drop=0.0).to(device)
opt_l1 = optim.Adam(m_l1.parameters(), lr=1e-3)
print("L1 penalty:")
_ = train_epochs(m_l1, opt_l1, train_dl, val_dl, epochs=5, l1_lambda=1e-6)

#6. Dropout
m_do = SmallCNN(p_drop=0.5).to(device)
opt_do = optim.Adam(m_do.parameters(), lr=1e-3)
print("Dropout 0.5:")
_ = train_epochs(m_do, opt_do, train_dl, val_dl, epochs=5)

#7. Data Augmentation
aug = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
train_aug = datasets.FashionMNIST('data', train=True, download=True, transform=aug)
train_dl_aug = DataLoader(train_aug, batch_size=128, shuffle=True)
m_aug = SmallCNN(p_drop=0.3).to(device)
opt_aug = optim.Adam(m_aug.parameters(), lr=1e-3, weight_decay=1e-4)
print("Augmentation + L2 + Dropout:")
_ = train_epochs(m_aug, opt_aug, train_dl_aug, val_dl, epochs=5)
