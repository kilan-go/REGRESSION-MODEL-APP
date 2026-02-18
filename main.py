import torch
import json
# import pydantic
# import requests
import torch.nn as nn
from flask import Flask, render_template, request, jsonify

# Model architecture
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,5)
        self.l2 = nn.Linear(5,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.l2(x)
        return x

model = RegressionModel()
model.load_state_dict(torch.load(f="Model/model.pth", weights_only=True))
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_value = float(data['input'])

    input_tensor = torch.tensor([[input_value]], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)