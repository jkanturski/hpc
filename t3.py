import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(42)

# --- synthetic dataset ---
N = 20000
X = torch.randn(N, 20)
y = (X.sum(dim=1) > 0).long()

# --- model ---
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- training params ---
epochs = 10
batch_size = 64

start = time.time()

for epoch in range(epochs):
    perm = torch.randperm(N)
    total_loss = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        xb = X[idx]
        yb = y[idx]

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss:.4f}")

end = time.time()
print(f"Training time: {end - start:.2f} seconds")
