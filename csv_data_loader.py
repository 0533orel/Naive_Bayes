import pandas as pd

class CsvDataLoader:
    def __init__(self, path: str, target_column: str):
        self.path = path
        self.target_column = target_column
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.path)

    def clean_data(self):
        if self.df is None:
            raise ValueError("the DataFrame is empty")
        self.df = self.df.dropna()
