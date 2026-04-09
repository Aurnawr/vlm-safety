import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

# Load dataset lengths to split features accurately
with open("/opt/watchdog/users/arnav/vlm-safety/datasets/dataset.json", "r", encoding="utf-8") as f:
    dataset1 = json.loads(f.read().strip())
with open("/opt/watchdog/users/arnav/vlm-safety/datasets/dataset2.json", "r", encoding="utf-8") as f:
    dataset2 = json.loads(f.read().strip())

n1 = len(dataset1)
n2 = len(dataset2)

# Load features from generation run
all_features = np.load('all_features.npy')

# Fit PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Load the refusal vector and project it
v_refusal = torch.load('/opt/watchdog/users/arnav/vlm-safety/v_refusal_last_layer.pt').cpu().float().numpy()
# To get the direction in PCA space without the mean shift offset:
# Project the vector itself onto the principal components
v_refusal_pca = pca.transform(pca.mean_.reshape(1, -1) + v_refusal.reshape(1, -1))[0] - pca.transform(pca.mean_.reshape(1, -1))[0]

# Normalize for visualization if you want a directional arrow, or scale it
# We'll scale it so it spans roughly the half-width of the scatter plot
max_val = np.max(np.abs(reduced_features))
v_refusal_plot = (v_refusal_pca / np.linalg.norm(v_refusal_pca)) * (max_val * 0.8)

# Split back to individual lists
h_text_pca = reduced_features[0 : n1]
b_text_pca = reduced_features[n1 : 2*n1]
h_image_pca = reduced_features[2*n1 : 2*n1 + n2]
b_image_pca = reduced_features[2*n1 + n2 : 2*n1 + 2*n2]

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(h_text_pca[:, 0], h_text_pca[:, 1], c='red', marker='o', label='Harmful (Dataset 1)')
plt.scatter(b_text_pca[:, 0], b_text_pca[:, 1], c='blue', marker='o', label='Benign (Dataset 1)')
plt.scatter(h_image_pca[:, 0], h_image_pca[:, 1], c='red', marker='x', label='Harmful (Dataset 2)')
plt.scatter(b_image_pca[:, 0], b_image_pca[:, 1], c='blue', marker='x', label='Benign (Dataset 2)')

# Plot refusal direction as an arrow originating from the center
plt.arrow(0, 0, v_refusal_plot[0], v_refusal_plot[1], 
          color='green', width=max_val*0.01, head_width=max_val*0.05, 
          length_includes_head=True, label='Refusal Direction')

plt.title('PCA of Last Layer Hidden States + Refusal Direction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('pca_with_refusal_direction.png')
print("Saved augmented PCA plot to pca_with_refusal_direction.png")