import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

N = 1_000_000
X = torch.randn(N, 100)
y = (X.sum(dim=1) > 0).long()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

model = nn.Sequential(
    nn.Linear(100, 512),
    nn.ReLU(),
    nn.Linear(512, 2),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

start = time.time()

for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")

print("Time:", time.time() - start)
