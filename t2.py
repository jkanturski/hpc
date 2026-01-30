import torch
import torch.nn as nn
import torch.optim as optim
import time

# Synthetic dataset
N = 200_000
X = torch.randn(N, 100)
y = (X.sum(dim=1) > 0).long()

# Simple model
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

torch.set_num_threads(int(torch.get_num_threads()))

start = time.time()

for epoch in range(5):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} | loss={loss.item():.4f}")

print("Time:", time.time() - start)
