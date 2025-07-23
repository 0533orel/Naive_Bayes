import pandas as pd
import os

class LoadData:
    """
    A class for loading and preprocessing CSV/Excel/JSON data for classification tasks.

    Attributes:
        __dataset (pd.DataFrame): The loaded dataset.
        __target_col (str): The name of the target (label) column.
    """

    def __init__(self, path: str, target_col: str = None):
        """
        Initializes the CsvDataLoader by reading a data file and extracting the target column.

        Args:
            path (str): The file path (CSV, Excel, or JSON).
            target_col (str, optional): The name of the target column.
                                        If not provided, the last column will be used.

        Raises:
            ValueError: If the file cannot be loaded or the target column is not found.
        """
        try:
            self.__dataset = self.__load_file(path)

            if self.__dataset.empty:
                raise ValueError("Loaded file is empty.")

            if target_col is None or target_col.strip() == "":
                self.__target_col = self.__dataset.columns[-1]
            else:
                if target_col not in self.__dataset.columns:
                    raise ValueError(f"Target column '{target_col}' not found in dataset.")
                self.__target_col = target_col

        except Exception as e:
            raise ValueError(f"\nProblem loading file: {e}")

    def __load_file(self, path: str) -> pd.DataFrame:
        """
        Loads the file into a pandas DataFrame based on file extension.

        Args:
            path (str): File path.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        ext = os.path.splitext(path)[-1].lower()

        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @property
    def dataset(self) -> pd.DataFrame:
        """Returns the loaded dataset."""
        return self.__dataset

    @property
    def target_col(self) -> str:
        """Returns the target column name."""
        return self.__target_col
