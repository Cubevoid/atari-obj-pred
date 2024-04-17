from src.data_collection.data_loader import DataLoader

if __name__ == "__main__":
    data_loader = DataLoader("Pong", "FastSAM-x", 32, 4)
    print(data_loader.sample(2, 10, "cpu"))
