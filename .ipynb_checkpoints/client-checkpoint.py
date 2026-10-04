from collections import OrderedDict
import os
import sqlite3
import time
import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

HOSPITAL_ID = os.getenv("HOSPITAL_ID", "hospital_A")
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "server:8080")
device = torch.device("cpu")

class DiagnosticNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 input features matched to the Kaggle Diabetes Dataset
        self.layer1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.output(self.relu(self.layer2(self.relu(self.layer1(x))))))

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_name):
        self.model = DiagnosticNet().to(device)
        db_path = f"privacare_dbs/{hospital_name}.db"
        
        # Load local data from the isolated SQLite database
        conn = sqlite3.connect(db_path)
        train_df = pd.read_sql("SELECT * FROM train_data", conn)
        test_df = pd.read_sql("SELECT * FROM test_data", conn)
        conn.close()

        # Extract raw features
        X_train_raw = train_df.drop(columns=["target"]).values
        X_test_raw = test_df.drop(columns=["target"]).values
        
        # Apply Feature Scaling (Normalization) based on training data
        train_mean = X_train_raw.mean(axis=0)
        train_std = X_train_raw.std(axis=0) + 1e-7
        
        X_train_norm = (X_train_raw - train_mean) / train_std
        X_test_norm = (X_test_raw - train_mean) / train_std

        # Convert to PyTorch Tensors
        self.X_train = torch.tensor(X_train_norm, dtype=torch.float32)
        self.y_train = torch.tensor(train_df["target"].values, dtype=torch.float32).view(-1, 1)
        self.X_test = torch.tensor(X_test_norm, dtype=torch.float32)
        self.y_test = torch.tensor(test_df["target"].values, dtype=torch.float32).view(-1, 1)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        self.model.train()
        # 15 local epochs for deeper pattern recognition on the large dataset
        for _ in range(15):
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()

        # Differential privacy: inject Gaussian noise prior to transmission
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.normal(mean=0.0, std=0.05, size=param.size())
                param.add_(noise)

        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.BCELoss()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            loss = criterion(outputs, self.y_test).item()
            preds = (outputs >= 0.5).float()
            accuracy = (preds == self.y_test).float().mean().item()
        return loss, len(self.X_test), {"accuracy": accuracy}

# Pause briefly to allow server gRPC startup
time.sleep(3)
print(f"[{HOSPITAL_ID}] Connecting to {SERVER_ADDRESS} via isolated Docker network...")
fl.client.start_client(
    server_address=SERVER_ADDRESS,
    client=HospitalClient(HOSPITAL_ID).to_client(),
)