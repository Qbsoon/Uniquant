import torch
import torch.nn as nn
import cma
import numpy as np


# -------------------------------------------------
# Your model
# -------------------------------------------------
class Model32to32(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(32, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 32)
		)

	def forward(self, x):
		return self.net(x)


# -------------------------------------------------
# Flatten / unflatten utils
# -------------------------------------------------
def get_flat_params(model):
	return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()

def set_flat_params(model, flat_params):
	flat = torch.tensor(flat_params, dtype=torch.float32)
	ptr = 0
	for p in model.parameters():
		n = p.numel()
		p.data[:] = flat[ptr:ptr+n].view_as(p)
		ptr += n


# -------------------------------------------------
# Your batch-processing callback
# -------------------------------------------------
def process_all_batches_fn(model):
	for batch in my_batches_32:
		batch = batch.to(device)
		output = model(batch)
		# You store / record / accumulate outputs here
		pass


# -------------------------------------------------
# CMA-ES fitness function
# -------------------------------------------------
def cma_objective(flat_params):
	# 1. load candidate weights
	set_flat_params(model, flat_params)

	# 2. run your batches (you do everything)
	process_all_batches_fn(model)

	# 3. get scalar loss (black-box)
	return float(loss_func())  # CMA expects plain float


# -------------------------------------------------
# TRAINING CALL (CMA-ES)
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model32to32().to(device)

# initial parameter vector
x0 = get_flat_params(model)

# step size (sigma)
sigma = 0.05

# CMA-ES options
opts = {
	'popsize': 64,         # population per generation
	'maxiter': 200,        # number of generations (epochs)
	'verb_disp': 1,
	'verb_log': 0,
}

# RUN CMA-ES
es = cma.CMAEvolutionStrategy(x0, sigma, opts)
es.optimize(cma_objective)

# best solution
best_params = es.result.xbest
set_flat_params(model, best_params)
torch.save(model.state_dict(), "model_cma_es.pt")

print("CMA-ES training finished.")
