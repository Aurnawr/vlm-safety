import torch
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def main():
    model_id = 'llava-hf/llava-1.5-7b-hf'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading {model_id} on {device}...')
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    blank_image = Image.new('RGB', (224, 224), color='white')

    def get_all_hidden_states(prompt, image):
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
        
        for k, v in inputs.items():
            if v.dtype == torch.float32:
                inputs[k] = v.to(device, dtype=torch.float16)
            else:
                inputs[k] = v.to(device)
                
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            # outputs.hidden_states[0] tuple has length 33 (embedding + 32 layers)
            hiddens = []
            for layer_idx in range(1, 33):
                layer_hidden_state = outputs.hidden_states[0][layer_idx]
                last_token_hidden = layer_hidden_state[:, -1, :].squeeze(0).cpu().float()
                hiddens.append(last_token_hidden)
            
        # Stack into shape (32, hidden_dim)
        return torch.stack(hiddens)

    def compute_refusal_direction(harmful_states, benign_states):
        H_plus = torch.stack(harmful_states)
        H_minus = torch.stack(benign_states)
        
        Delta_H = H_plus - H_minus
        mean_diff = torch.mean(Delta_H, dim=0)
        Delta_H_centered = Delta_H - mean_diff
        
        U, S, V = torch.svd(Delta_H_centered)
        v_refusal = V[:, 0]
        return v_refusal / torch.norm(v_refusal)

    # Resolve paths reliably
    base_dir = "/opt/watchdog/users/arnav/vlm-safety"
    ds1_path = os.path.join(base_dir, "datasets/dataset.json")
    ds2_path = os.path.join(base_dir, "datasets/dataset2.json")

    # Process Dataset 1
    with open(ds1_path, "r", encoding="utf-8") as f:
        dataset1 = json.loads(f.read().strip())

    harmful_hidden_1 = []
    benign_hidden_1 = []
    print("Processing dataset 1...")
    for item in dataset1:
        harmful_hidden_1.append(get_all_hidden_states(item['harmful_prompt'], blank_image))
        benign_hidden_1.append(get_all_hidden_states(item['benign_prompt'], blank_image))
        
    # Process Dataset 2
    with open(ds2_path, "r", encoding="utf-8") as f:
        dataset2 = json.loads(f.read().strip())

    harmful_hidden_2 = []
    benign_hidden_2 = []
    print("Processing dataset 2...")
    for item in dataset2:
        harmful_hidden_2.append(get_all_hidden_states(item['harmful_prompt'], blank_image))
        benign_hidden_2.append(get_all_hidden_states(item['benign_prompt'], blank_image))

    # Convert to tensors of shape (num_samples, 32, hidden_dim)
    harmful_hidden_1_tensor = torch.stack(harmful_hidden_1)
    benign_hidden_1_tensor = torch.stack(benign_hidden_1)
    harmful_hidden_2_tensor = torch.stack(harmful_hidden_2)
    benign_hidden_2_tensor = torch.stack(benign_hidden_2)

    os.makedirs(os.path.join(base_dir, "generation_scripts", "pca_plots"), exist_ok=True)
    
    print("Computing PCA and Refusal Directions per layer...")
    # Loop efficiently over all extracted layers
    for layer_idx in range(32):
        layer_to_probe = layer_idx + 1
        
        # Extract slices properly (shape: (num_samples, hidden_dim)) for the current layer
        h1 = list(harmful_hidden_1_tensor[:, layer_idx, :])
        b1 = list(benign_hidden_1_tensor[:, layer_idx, :])
        h2 = list(harmful_hidden_2_tensor[:, layer_idx, :])
        b2 = list(benign_hidden_2_tensor[:, layer_idx, :])

        v_refusal_1 = compute_refusal_direction(h1, b1)
        v_refusal_2 = compute_refusal_direction(h2, b2)

        # Calculate Cosine Similarity
        cosine_sim = F.cosine_similarity(v_refusal_1.unsqueeze(0), v_refusal_2.unsqueeze(0)).item()
        print(f"Layer {layer_to_probe} - Cosine Sim: {cosine_sim:.4f}")

        # Combine array elements for PCA
        h1_np = [t.numpy() for t in h1]
        b1_np = [t.numpy() for t in b1]
        h2_np = [t.numpy() for t in h2]
        b2_np = [t.numpy() for t in b2]

        all_features = np.array(h1_np + b1_np + h2_np + b2_np)
        
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(all_features)

        v_refusal_1_np = v_refusal_1.numpy()
        v_refusal_2_np = v_refusal_2.numpy()
        
        mean_proj = pca.transform(pca.mean_.reshape(1, -1))[0]
        v1_pca = pca.transform(pca.mean_.reshape(1, -1) + v_refusal_1_np.reshape(1, -1))[0] - mean_proj
        v2_pca = pca.transform(pca.mean_.reshape(1, -1) + v_refusal_2_np.reshape(1, -1))[0] - mean_proj

        max_val = np.max(np.abs(reduced_features))
        v1_plot = (v1_pca / np.linalg.norm(v1_pca)) * (max_val * 0.8)
        v2_plot = (v2_pca / np.linalg.norm(v2_pca)) * (max_val * 0.8)

        n1 = len(dataset1)
        n2 = len(dataset2)

        # Plot everything
        plt.figure(figsize=(10, 8))
        
        plt.scatter(reduced_features[0:n1, 0], reduced_features[0:n1, 1], c='red', marker='o', label='Harmful (Dataset 1)')
        plt.scatter(reduced_features[n1:2*n1, 0], reduced_features[n1:2*n1, 1], c='blue', marker='o', label='Benign (Dataset 1)')
        
        start2 = 2*n1
        plt.scatter(reduced_features[start2:start2+n2, 0], reduced_features[start2:start2+n2, 1], c='darkred', marker='x', label='Harmful (Dataset 2)')
        plt.scatter(reduced_features[start2+n2:, 0], reduced_features[start2+n2:, 1], c='darkblue', marker='x', label='Benign (Dataset 2)')

        plt.arrow(0, 0, v1_plot[0], v1_plot[1], color='green', width=max_val*0.01, head_width=max_val*0.03, length_includes_head=True, label='Refusal Dir (Dataset 1)')
        plt.arrow(0, 0, v2_plot[0], v2_plot[1], color='purple', width=max_val*0.01, head_width=max_val*0.03, length_includes_head=True, label='Refusal Dir (Dataset 2)')

        plt.title(f'PCA of Layer {layer_to_probe} Hidden States + Refusal Directions\nCosine Similarity: {cosine_sim:.4f}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        
        out_file = os.path.join(base_dir, "generation_scripts", "pca_plots", f'combined_pca_layer_{layer_to_probe:02d}.png')
        plt.savefig(out_file)
        plt.close() # Close memory handle!
        
    print(f"Generated all 32 plots successfully natively!")

if __name__ == "__main__":
    main()