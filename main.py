import torch
import json
from PIL import Image
import numpy as np 
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id= "llava-hf/llava-1.5-7b-hf"
device= 'cuda' if torch.cuda.is_available() else 'cpu'
layer_to_probe= 10 # probing the final alyer 

print (f'Loading{model_id} on {device}...')
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)
model.eval()

#create the control blank image 
blank_image= Image.new('RGB', (224,224), color = 'white')

def get_last_token(prompt, image):

    formatted_prompt= f"USER: <image>\n{prompt}\n ASSISTANT:"
    inputs = processor ( text=formatted_prompt, images= image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states= True)

    hidden_states= outputs.hidden_states
    #hidden states= (layers, batch_size, seq_length, hidden_dim )
    last_layer_token= hidden_states[layer_to_probe]
    last_token_state= last_layer_token[0,-1,:]

    return last_token_state.cpu().to(torch.float32)

harmful_activations=[]
benign_activations= []

import json

with open("/opt/watchdog/users/arnav/vlm-safety/datasets/dataset.json", "r", encoding="utf-8") as f:
    content = f.read().strip()

dataset = json.loads(content)

print ('Caching activations...')
for item in dataset:
    h_state= get_last_token(item['harmful_prompt'], blank_image)
    b_state= get_last_token(item['benign_prompt'],blank_image)

    harmful_activations.append(h_state)
    benign_activations.append(b_state)

H_plus= torch.stack(harmful_activations)
H_minus= torch.stack(benign_activations)

#Calculating the refusal direction

print("Calculating refusal direction")

#Delta_H (N, hidden_dim)
Delta_H= H_plus -H_minus

mean_diff= torch.mean(Delta_H,dim=0)
Delta_H_centered= Delta_H - mean_diff

#svd to compute the principal refusal direction
U,S,V = torch.svd(Delta_H_centered)

# V shape: (hidden_dim, hidden_dim) -> V[:, 0] shape: (hidden_dim,)
v_refusal = V[:, 0]

v_refusal= v_refusal/torch.norm(v_refusal)


print(f"Extraction complete. Refusal vector shape: {v_refusal.shape}")
print(f"L2 Norm of vector: {torch.norm(v_refusal).item()}")

# Save the vector for future steering/intervention experiments
torch.save(v_refusal, "v_refusal_blank_image_layer_10.pt")
print("Saved v_refusal tensor to disk.")


