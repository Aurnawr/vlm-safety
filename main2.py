import torch
import json
from PIL import Image
import numpy as np 
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id= "llava-hf/llava-1.5-7b-hf'
device= 'cuda' if torch.cuda.is_available() else 'cpu'
