import pandas as pd

class CsvDataLoader:
    def __init__(self, path: str):
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            self.df = None
            raise ValueError(f"Problem loading file: {e}")

    def clean_data(self):
        if self.df is None:
            raise ValueError("The DataFrame is empty.")

        mask = self.df.columns.str.contains('id', case=False)
        self.df = self.df.loc[:, ~mask]

        self.df = self.df.dropna()
