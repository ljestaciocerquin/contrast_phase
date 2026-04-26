import torch
from merlin import Merlin

# ---- 1. Initialize model ----
model = Merlin(ImageEmbedding=True)
model.eval()

# ---- 2. Create dummy input ----
# Typical 3D CT shape: (B, C, D, H, W)
B, C, D, H, W = 2, 1, 128, 128, 128

x = torch.randn(B, C, D, H, W)

print("Input shape:", x.shape)

# ---- 3. Forward pass ----
with torch.no_grad():
    out = model(x)

# ---- 4. Inspect output ----
print("\nType of output:", type(out))

if isinstance(out, dict):
    print("\nKeys:", out.keys())
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")

elif torch.is_tensor(out):
    print("Output shape:", out.shape)

else:
    print("Output:", out)