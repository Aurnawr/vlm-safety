import torch
import matplotlib.pyplot as plt
import numpy as np


x= [0.9940,0.9778,0.9579,0.9289,0.9635,0.9467,0.9283,0.8395,0.9383,0.9286,0.9163,0.9420,0.9501,0.9413,0.9403,0.9442,0.8524,0.8781,0.8787,0.8828,0.8463,0.8027,0.8351,0.7174,0.7174,0.8135,0.8226,0.8358,0.8513, 0.8504, 0.8515, 0.8638]
y=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

plt.plot(y,x)
plt.ylabel('cosine similarity between the refusal directions of the two datasets')
plt.xlabel('layers')
plt.savefig('plot of cosine similarities.png')
plt.show()