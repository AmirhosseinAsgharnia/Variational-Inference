# we are going to model the following:
# y = sin(x) + e(x); e(x) ~ N(0, \sigma(x)^2)
# \sigma(x) = 0.1 + 0.4 . 1_{|x|>\pi/2}

import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

os.system("cls")

#%% Function

N = 100
x = torch.linspace(-torch.pi, torch.pi, N).unsqueeze(1)
sigma = 0.1 * torch.ones(N ,1)

for i in range(N):
    if x[i].abs() > torch.pi/2:
        sigma[i] += 0.4 

eps = sigma * torch.randn_like(x) 
y_ideal = torch.sin(x)
y_meas  = y_ideal + eps

#%% Class

class Gaussian_Linear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        self.register_buffer("w_mu", torch.zeros(inputs, outputs))
        self.register_buffer("w_std", torch.ones(inputs, outputs))
        self.register_buffer("b_mu", torch.zeros(1, outputs))
        self.register_buffer("b_std", torch.ones(1, outputs))

    def forward(self, x):
        eps_w = torch.randn_like(self.w_mu) #type: ignore
        eps_b = torch.randn_like(self.b_mu) #type: ignore

        w = self.w_mu + eps_w * self.w_std #type: ignore
        b = self.b_mu + eps_b * self.b_std #type: ignore

        return x @ w + b


#%%

fig, axe = plt.subplots(1, 2, figsize=(10, 4))
axe[0].plot(x, y_ideal, color = 'r')
axe[0].scatter(x, y_meas, color = 'b')
axe[0].set_xlabel("x")
axe[0].set_ylabel("y")
axe[0].minorticks_on()
axe[0].grid(True, which="major", linestyle="-", linewidth=0.8)
axe[0].grid(True, which="minor", linestyle=":", linewidth=0.5)
plt.show()