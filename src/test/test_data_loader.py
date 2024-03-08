from src.data_collection.data_loader import DataLoader

if __name__ == "__main__":
    data_loader = DataLoader("Pong")
    print(data_loader.sample(2, 1))
