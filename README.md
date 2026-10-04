# PrivaCare: Federated Medical AI
PrivaCare is a privacy-preserving federated learning platform. It enables isolated hospital nodes to collaboratively train a PyTorch diagnostic neural network on 100,000+ patient records without ever centralizing or sharing raw clinical data.

## System Architecture
* **Data Sovereignty:** Clinical records remain locked inside isolated local SQLite databases.
* **Federated Learning:** A central orchestrator (Flower) aggregates local model weights using the `FedAvg` strategy.
* **Differential Privacy:** Gaussian noise is injected into local model parameters prior to transmission to mathematically prevent model-inversion attacks.
* **Infrastructure:** The entire network operates across isolated Docker containers communicating securely over a private virtual bridge network.

## Execution
To launch the decentralized hospital network:
1. Ensure Docker Desktop is running.
2. Execute the build command:
docker compose up --build