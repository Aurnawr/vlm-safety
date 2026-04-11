#this script is to check how well the model refuses harmful queries when its attention is on the image vs only on the prompt. 

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F

model_id = 'llava-hf/llava-1.5-7b-hf'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_to_probe = 10
alpha = 1.0
print('Loading the model and vector...')
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

#create the control blank image 
blank_image = Image.new('RGB', (224,224), color = 'white')

def generation(prompt, image, layer=-1):
    formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
    
    # ensure proper types
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(device, dtype=torch.float16)
        else:
            inputs[k] = v.to(device)
            
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,  # We just need the hidden state of the first generated token
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
        # outputs.hidden_states is a tuple of tuples: (generated_token, layer)
        # We take the first generated token, specified layer, last token in sequence
        # Shape is (batch_size, sequence_length, hidden_size)
        layer_hidden_state = outputs.hidden_states[0][layer]
        last_token_hidden = layer_hidden_state[:, -1, :].squeeze(0).cpu().float()
        
    return last_token_hidden

def compute_refusal_direction(harmful_states, benign_states):
    H_plus = torch.stack(harmful_states)
    H_minus = torch.stack(benign_states)
    
    Delta_H = H_plus - H_minus
    mean_diff = torch.mean(Delta_H, dim=0)
    Delta_H_centered = Delta_H - mean_diff
    
    U, S, V = torch.svd(Delta_H_centered)
    v_refusal = V[:, 0]
    return v_refusal / torch.norm(v_refusal)

#actual dataset loading
with open("/opt/watchdog/users/arnav/vlm-safety/datasets/dataset.json", "r", encoding="utf-8") as f:
    dataset = json.loads(f.read().strip())

harmful_hidden_text=[]
benign_hidden_text=[]
print("Processing dataset 1...")
# Note: Since the prompt mentions "dataset1 and 2", I'm assuming dataset1 has blank image
for item in dataset:
    h_hidden = generation(item['harmful_prompt'], blank_image, layer=layer_to_probe)
    b_hidden = generation(item['benign_prompt'], blank_image, layer=layer_to_probe)
    harmful_hidden_text.append(h_hidden)
    benign_hidden_text.append(b_hidden)

# Calculate refusal vector for dataset 1
v_refusal_1 = compute_refusal_direction(harmful_hidden_text, benign_hidden_text)

#actual dataset loading for 2
with open("/opt/watchdog/users/arnav/vlm-safety/datasets/dataset2.json", "r", encoding="utf-8") as f:
    dataset2 = json.loads(f.read().strip())

harmful_hidden_image = []
benign_hidden_image = []

print("Processing dataset 2...")
for item in dataset2:
    h_hidden = generation(item['harmful_prompt'], blank_image, layer=layer_to_probe)
    b_hidden = generation(item['benign_prompt'], blank_image, layer=layer_to_probe)
 
    harmful_hidden_image.append(h_hidden)
    benign_hidden_image.append(b_hidden)

# Calculate refusal vector for dataset 2
v_refusal_2 = compute_refusal_direction(harmful_hidden_image, benign_hidden_image)

# Calculate cosine similarity
cosine_sim = F.cosine_similarity(v_refusal_1.unsqueeze(0), v_refusal_2.unsqueeze(0)).item()
print(f"Cosine Similarity between Refusal Directions: {cosine_sim:.4f}")

# Convert to numpy for PCA
harmful_hidden_text_np = [t.numpy() for t in harmful_hidden_text]
benign_hidden_text_np = [t.numpy() for t in benign_hidden_text]
harmful_hidden_image_np = [t.numpy() for t in harmful_hidden_image]
benign_hidden_image_np = [t.numpy() for t in benign_hidden_image]

# Perform PCA Visualization
all_features = np.array(
    harmful_hidden_text_np + 
    benign_hidden_text_np + 
    harmful_hidden_image_np + 
    benign_hidden_image_np
)
np.save(f'all_features_{layer_to_probe}.npy', all_features)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Project both refusal vectors onto PCA
v_refusal_1_np = v_refusal_1.numpy()
v_refusal_2_np = v_refusal_2.numpy()
v_refusal_1_pca = pca.transform(pca.mean_.reshape(1, -1) + v_refusal_1_np.reshape(1, -1))[0] - pca.transform(pca.mean_.reshape(1, -1))[0]
v_refusal_2_pca = pca.transform(pca.mean_.reshape(1, -1) + v_refusal_2_np.reshape(1, -1))[0] - pca.transform(pca.mean_.reshape(1, -1))[0]

# Normalize for plotting arrows
max_val = np.max(np.abs(reduced_features))
v_refusal_1_plot = (v_refusal_1_pca / np.linalg.norm(v_refusal_1_pca)) * (max_val * 0.8)
v_refusal_2_plot = (v_refusal_2_pca / np.linalg.norm(v_refusal_2_pca)) * (max_val * 0.8)

# Split back to individual lists
n1 = len(dataset)
n2 = len(dataset2)

h_text_pca = reduced_features[0 : n1]
b_text_pca = reduced_features[n1 : 2*n1]
h_image_pca = reduced_features[2*n1 : 2*n1 + n2]
b_image_pca = reduced_features[2*n1 + n2 : 2*n1 + 2*n2]

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(h_text_pca[:, 0], h_text_pca[:, 1], c='red', marker='o', label='Harmful (Dataset 1)')
plt.scatter(b_text_pca[:, 0], b_text_pca[:, 1], c='blue', marker='o', label='Benign (Dataset 1)')
plt.scatter(h_image_pca[:, 0], h_image_pca[:, 1], c='darkred', marker='x', label='Harmful (Dataset 2)')
plt.scatter(b_image_pca[:, 0], b_image_pca[:, 1], c='darkblue', marker='x', label='Benign (Dataset 2)')

# Plot arrows
plt.arrow(0, 0, v_refusal_1_plot[0], v_refusal_1_plot[1], 
          color='green', width=max_val*0.01, head_width=max_val*0.03, 
          length_includes_head=True, label='Refusal Dir (Dataset 1)')
plt.arrow(0, 0, v_refusal_2_plot[0], v_refusal_2_plot[1], 
          color='purple', width=max_val*0.01, head_width=max_val*0.03, 
          length_includes_head=True, label='Refusal Dir (Dataset 2)')

plt.title(f'PCA of Layer {layer_to_probe} Hidden States + Refusal Directions\nSim: {cosine_sim:.4f}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig(f'pca_visualization_layer_abs{abs(layer_to_probe)}.png')
print(f"Saved PCA plot to pca_visualization_layer_abs{abs(layer_to_probe)}.png")

