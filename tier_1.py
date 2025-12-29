# we are going to model the following:
# y = sin(x) + e(x); e(x) ~ N(0, \sigma(x)^2)
# \sigma(x) = 0.1 + 0.4 . 1_{|x|>\pi/2}

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

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

#%% Classes

class Gaussian_Linear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        self.w_mu = nn.Parameter(torch.zeros(inputs, outputs))
        self.w_rho = nn.Parameter(torch.full((inputs, outputs), -3.0))
        self.b_mu = nn.Parameter(torch.zeros(1, outputs))
        self.b_rho = nn.Parameter(torch.full((1, outputs), -3.0))

        self.register_buffer("w_mu_prior" ,torch.zeros(inputs, outputs))
        self.register_buffer("w_sigma_prior", torch.ones(inputs, outputs))
        self.register_buffer("b_mu_prior", torch.zeros(1, outputs))
        self.register_buffer("b_sigma_prior", torch.ones(1, outputs))

    def forward(self, x):

        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        w = self.w_mu + w_sigma * torch.randn_like(self.w_mu)
        b = self.b_mu + b_sigma * torch.randn_like(self.b_mu)

        return x @ w + b
    
    def kl_divergence(self):

        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        kl_w = 0.5 * (torch.log(self.w_sigma_prior ** 2/ w_sigma ** 2) + (w_sigma ** 2 + (self.w_mu - self.w_mu_prior) ** 2) / self.w_sigma_prior ** 2 - 1).sum()
        kl_b = 0.5 * (torch.log(self.b_sigma_prior ** 2/ b_sigma ** 2) + (b_sigma ** 2 + (self.b_mu - self.b_mu_prior) ** 2) / self.b_sigma_prior ** 2 - 1).sum()

        return kl_b + kl_w
    
class BNN(nn.Module):
    def __init__(self, hidden = 10) -> None:
        super().__init__()

        self.fc1 = Gaussian_Linear(1, hidden)
        self.fc2 = Gaussian_Linear(hidden, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)

        return x


model = BNN(hidden=20).to(device)
x, y_meas, sigma = x.to(device), y_meas.to(device), sigma.to(device)

def gaussian_nll(y, yhat, sigma):
    return ((y - yhat)**2 / (2*sigma**2) + torch.log(sigma)).sum()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

max_epoch = 5000

for epoch in range(max_epoch):
    model.train()
    y_pred = model(x)

    kl = model.fc1.kl_divergence() + model.fc2.kl_divergence()
    nll = gaussian_nll(y_meas, y_pred, sigma)

    loss = (nll + kl) / N

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch % 10) == 0:
        print(f"Epoch = {epoch}, Loss = {loss}, KL = {kl}")
        
model.eval()
y_pred = model(x)
#%% Plotting

fig, axe = plt.subplots(1, 2, figsize=(10, 4))
axe[0].plot(x, y_ideal, color = 'r')
axe[0].scatter(x, y_meas, color = 'b')
axe[0].plot(x, y_pred.detach(), color = 'g')
axe[0].set_xlabel("x")
axe[0].set_ylabel("y")
axe[0].minorticks_on()
axe[0].grid(True, which="major", linestyle="-", linewidth=0.8)
axe[0].grid(True, which="minor", linestyle=":", linewidth=0.5)
plt.show()