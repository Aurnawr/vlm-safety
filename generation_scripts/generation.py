#this script is to check how well the model refuses harmful queries when its attention is on the image vs only on the prompt. 

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model_id = 'llava-hf/llava-1.5-7b-hf'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_to_probe = -1 
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

def generation(prompt, image):
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
            max_new_tokens=500,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
        # Decode the newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_text = processor.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True)[0]
        print("Generated:", generated_text)

        # outputs.hidden_states is a tuple of tuples: (generated_token, layer)
        # We take the first generated token, last layer, last token in sequence
        # Shape is (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.hidden_states[0][-1]
        last_token_hidden = last_hidden_state[:, -1, :].squeeze(0).cpu().float().numpy()
        
    return generated_text, last_token_hidden

#actual dataset loading
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.loads(f.read().strip())

harmful_generation_text = []
benign_generation_text = []
harmful_hidden_text = []
benign_hidden_text = []

print("Processing dataset 1...")
# Note: Since the prompt mentions "dataset1 and 2", I'm assuming dataset1 has blank image
for item in dataset:
    h_text, h_hidden = generation(item['harmful_prompt'], blank_image)
    b_text, b_hidden = generation(item['benign_prompt'], blank_image)
    harmful_generation_text.append(h_text)
    benign_generation_text.append(b_text)
    harmful_hidden_text.append(h_hidden)
    benign_hidden_text.append(b_hidden)

#actual dataset loading for 2
with open("dataset2.json", "r", encoding="utf-8") as f:
    dataset2 = json.loads(f.read().strip())
    
harmful_generation_image = []
benign_generation_image = []
harmful_hidden_image = []
benign_hidden_image = []

print("Processing dataset 2...")
for item in dataset2:
    h_text, h_hidden = generation(item['harmful_prompt'], blank_image)
    b_text, b_hidden = generation(item['benign_prompt'], blank_image)
    harmful_generation_image.append(h_text)
    benign_generation_image.append(b_text)
    harmful_hidden_image.append(h_hidden)
    benign_hidden_image.append(b_hidden)

# Perform PCA Visualization
all_features = np.array(
    harmful_hidden_text + 
    benign_hidden_text + 
    harmful_hidden_image + 
    benign_hidden_image
)

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

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
plt.scatter(h_image_pca[:, 0], h_image_pca[:, 1], c='red', marker='x', label='Harmful (Dataset 2)')
plt.scatter(b_image_pca[:, 0], b_image_pca[:, 1], c='blue', marker='x', label='Benign (Dataset 2)')

plt.title('PCA of Last Layer Hidden States')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('pca_visualization.png')
print("Saved PCA plot to pca_visualization.png")

with open('generated_responses_hi.txt', 'w') as f:
    f.write('\n'.join(harmful_generation_image))

with open('generated_responses_ht.txt', 'w') as f:
    f.write('\n'.join(harmful_generation_text))

with open('generated_responses_bi.txt', 'w') as f:
    f.write('\n'.join(benign_generation_image))

with open('generated_responses_bt.txt', 'w') as f:
    f.write('\n'.join(benign_generation_text))

