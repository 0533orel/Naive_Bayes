import pandas as pd

class CsvDataLoader:
    def __init__(self, path: str):
        self.path = path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.path)

    def clean_data(self):
        if self.df is None:
            raise ValueError("the DataFrame is empty")
        mask = self.df.columns.str.contains('id', case=False, regex=True)
        self.df = self.df.loc[:, ~mask]
        self.df = self.df.dropna()
