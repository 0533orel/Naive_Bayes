import pandas as pd

class CsvDataLoader:
    def __init__(self, path: str, target_col: str):
        try:
            self.df = pd.read_csv(path)
            self.target_col = target_col
        except Exception as e:
            raise ValueError(f"Problem loading file: {e}")

        self.clean_data()

    def clean_data(self):
        if self.df is None:
            raise ValueError("The DataFrame is empty.")

        mask = self.df.columns.str.contains('index|id', case=False)
        self.df = self.df.loc[:, ~mask]

        self.df = self.df.dropna()

        self.df = self.df.loc[:, ~self.df.T.duplicated()]

        if self.target_col not in self.df.columns:
            raise ValueError(f"'{self.target_col}' column not found in DataFrame.")

        if self.df.columns[-1] != self.target_col:
            cols = [c for c in self.df.columns if c != self.target_col] + [self.target_col]
            self.df = self.df[cols]
