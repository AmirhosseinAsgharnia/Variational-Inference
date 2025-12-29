# we are going to model the following:
# y = x^3 - 0.5 x ^ 2

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
x = torch.linspace(-1, 1, N).unsqueeze(1)
sigma = torch.tensor(0.1)
eps = sigma * torch.randn_like(x) 
y_ideal = x ** 3 - 0.5 * x ** 2
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
    def __init__(self, hidden = 20) -> None:
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

optimizer = optim.Adam(model.parameters(), lr=1e-2)

max_epoch = 5000

for epoch in range(max_epoch):
    model.train()
    y_pred = model(x)

    kl = model.fc1.kl_divergence() + model.fc2.kl_divergence()
    S = 5
    nll = torch.stack([gaussian_nll(y_meas, model(x), sigma) for _ in range(S)]).mean()
    loss = (nll + kl) / N

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch % 10) == 0:
        print(f"Epoch = {epoch}, Loss = {loss}, KL = {kl}")

model.eval()
y_pred_1 = model(x)

#%% Evaluation (Chat wrote this part, not me)

@torch.no_grad()
def mc_predict(model, x, S=200):
    model.eval()  # doesn't stop sampling in your code, but keeps other layers consistent
    preds = []
    for _ in range(S):
        preds.append(model(x))          # (N,1)
    preds = torch.stack(preds, dim=0)   # (S,N,1)
    mean = preds.mean(dim=0)            # (N,1)
    std  = preds.std(dim=0)             # (N,1)  epistemic
    q05  = preds.quantile(0.05, dim=0)  # (N,1)
    q95  = preds.quantile(0.95, dim=0)  # (N,1)
    return mean, std, q05, q95

mean, std, q05, q95 = mc_predict(model, x, S=300)

#%% Plotting

x_cpu = x.squeeze().detach().cpu().numpy()
mean_cpu = mean.squeeze().cpu().numpy()
q05_cpu  = q05.squeeze().cpu().numpy()
q95_cpu  = q95.squeeze().cpu().numpy()

plt.figure(figsize=(7,5))
plt.plot(x_cpu, y_ideal.squeeze().cpu().numpy(), 'r', label='true')
plt.scatter(x_cpu, y_meas.squeeze().cpu().numpy(), s=10, label='data')
plt.plot(x_cpu, mean_cpu, 'g', label='MC mean')
plt.fill_between(x_cpu, q05_cpu, q95_cpu, alpha=0.2, label='90% epistemic band')
plt.legend()
plt.grid(True)
plt.show()
