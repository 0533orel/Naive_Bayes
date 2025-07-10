import pandas as pd

class CsvDataLoader:
    def __init__(self, path: str, target_col: str = None):
        try:
            self._df = pd.read_csv(path)

            if target_col is None or target_col.strip() == "":
                self._target_col = self._df.columns[-1]
            else:
                target_col = rf"{target_col}"
                self._target_col = target_col

            self.clean_data()

        except Exception as e:
            raise ValueError(f"\nProblem loading file: {e}")

    def clean_data(self):
        if self._df is None:
            raise ValueError("\nThe DataFrame is empty.")

        mask = self._df.columns.str.contains('index|id', case=False)
        self._df = self._df.loc[:, ~mask]

        self._df = self._df.dropna()

        self._df = self._df.loc[:, ~self._df.T.duplicated()]

        if self._target_col not in self._df.columns:
            raise ValueError(f"\n'{self._target_col}' column not found in DataFrame.")

        if self._df.columns[-1] != self._target_col:
            cols = [c for c in self._df.columns if c != self._target_col] + [self._target_col]
            self._df = self._df[cols]
