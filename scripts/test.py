import sys
import importlib
import subprocess
import torch
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
import sklearn
import streamlit as st
import matplotlib.pyplot as plt

def check_package_version(package_name, expected_version):
    """Check if a package is installed and matches the expected version."""
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', 'Unknown')
        if installed_version != expected_version:
            print(f"Warning: {package_name} version mismatch! Expected {expected_version}, but found {installed_version}.")
        else:
            print(f"{package_name} is installed correctly (v{installed_version}).")
    except ImportError:
        print(f"Error: {package_name} is not installed.")
        print(f"Installing {package_name}...")
        subprocess.run([sys.executable, "-m", "pip", "install", f"{package_name}=={expected_version}"])

# Package list with expected versions
packages = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",
    "torch_geometric": "2.6.1",
    "torch_scatter": "2.1.2",
    "torch_sparse": "0.6.18",
    "pyyaml": "6.0.2",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "matplotlib": "3.10.1",
    "scikit-learn": "1.3.0",
    "networkx": "3.2.1",
    "streamlit": "1.44.0",
}

# Check all packages
for package, version in packages.items():
    check_package_version(package, version)

# Simple functionality tests
def run_basic_tests():
    """Run basic tests to check compatibility."""
    try:
        print("\nRunning basic tests...")
        
        # Torch Test
        print("Checking PyTorch...")
        print("Torch CUDA available:", torch.cuda.is_available())
        tensor = torch.tensor([1.0, 2.0, 3.0])
        print("Torch Tensor:", tensor)
        
        # NumPy Test
        print("Checking NumPy...")
        array = np.array([1, 2, 3])
        print("NumPy Array:", array)
        
        # Pandas Test
        print("Checking Pandas...")
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        print("Pandas DataFrame:\n", df)
        
        # NetworkX Test
        print("Checking NetworkX...")
        G = nx.Graph()
        G.add_edge(1, 2)
        print("NetworkX Graph Nodes:", G.nodes())
        
        # Matplotlib Test
        print("Checking Matplotlib...")
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.title("Test Plot")
        plt.show()
        
        # Scikit-learn Test
        print("Checking Scikit-learn...")
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        print("Scikit-learn LinearRegression initialized successfully.")
        
        # Torch Geometric Test
        print("Checking Torch Geometric...")
        from torch_geometric.data import Data
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).T
        x = torch.tensor([[1], [2], [3]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        print("Torch Geometric Data:", data)
        
        # Streamlit Test
        print("Checking Streamlit...")
        print("Streamlit version:", st.__version__)
        print("All tests passed successfully!")
    except Exception as e:
        print("Error during tests:", e)

# Run the tests
run_basic_tests()