from pipelines.training_pipeline import train_pipeline
from zenml.client import Client # type: ignore

path = r'.\data\olist_customers_dataset.csv'

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    
    train_pipeline(path=path) # run pipeline
