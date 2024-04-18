import torch
import pickle

# Step 1: Load the .pkl file
with open('/home/sd/barc_data/trainingData/models/aggressive_blocking.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Assume `model_data` is a dictionary that matches the expected state dictionary of the model
# Step 2: Define the PyTorch model (ensure this matches the structure expected by the state dict)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Linear(10, 20)
        # Add other layers as per the model

    def forward(self, x):
        return self.layer1(x)

model = MyModel()

# Step 3: Load the state dictionary into the model
model.load_state_dict(model_data)

# Step 4: Save as a .pt file
torch.save(model.state_dict(), 'aggressive_blocking.pt')
