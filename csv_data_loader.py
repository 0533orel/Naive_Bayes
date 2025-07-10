import pandas as pd

class CsvDataLoader:
    """
    A class for loading and preprocessing CSV data for classification tasks.

    Attributes:
        __df (pd.DataFrame): The cleaned DataFrame loaded from the CSV file.
        __target_col (str): The name of the target (label) column.
    """

    def __init__(self, path: str, target_col: str = None):
        """
        Initializes the CsvDataLoader by reading a CSV file and cleaning the data.

        Args:
            path (str): The file path to the CSV file.
            target_col (str, optional): The name of the target column.
                                        If not provided or empty, the last column will be used.

        Raises:
            ValueError: If the file cannot be loaded or the target column is not found.
        """
        try:
            self.__df = pd.read_csv(path)

            if target_col is None or target_col.strip() == "":
                self.__target_col = self.__df.columns[-1]
            else:
                target_col = rf"{target_col}"
                self.__target_col = target_col

            self.clean_data()

        except Exception as e:
            raise ValueError(f"\nProblem loading file: {e}")

    @property
    def df(self):
        """
        Returns the cleaned DataFrame.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        return self.__df

    def clean_data(self):
        """
        Cleans the loaded DataFrame by:
        - Removing columns containing 'index' or 'id'
        - Dropping rows with missing values
        - Removing duplicate columns
        - Ensuring the target column is last

        Raises:
            ValueError: If the DataFrame is empty or the target column is missing.
        """
        if self.__df is None:
            raise ValueError("\nThe DataFrame is empty.")

        mask = self.__df.columns.str.contains('index|id', case=False)
        self.__df = self.__df.loc[:, ~mask]

        self.__df = self.__df.dropna()

        self.__df = self.__df.loc[:, ~self.__df.T.duplicated()]

        if self.__target_col not in self.__df.columns:
            raise ValueError(f"\n'{self.__target_col}' column not found in DataFrame.")

        if self.__df.columns[-1] != self.__target_col:
            cols = [c for c in self.__df.columns if c != self.__target_col] + [self.__target_col]
            self.__df = self.__df[cols]
