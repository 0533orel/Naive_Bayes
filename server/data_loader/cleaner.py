class Cleaner:
    def __init__(self, data, target):
        self.__dataset = data
        self.__target_col = target
        self.clean_data()

    @property
    def dataset(self):
        """
        Returns the cleaned DataFrame.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        return self.__dataset

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
        if self.__dataset is None:
            raise ValueError("\nThe DataFrame is empty.")

        mask = self.__dataset.columns.str.contains('index|id', case=False)
        self.__dataset = self.__dataset.loc[:, ~mask]

        self.__dataset = self.__dataset.dropna()

        self.__dataset = self.__dataset.loc[:, ~self.__dataset.T.duplicated()]

        if self.__target_col not in self.__dataset.columns:
            raise ValueError(f"\n'{self.__target_col}' column not found in DataFrame.")

        if self.__dataset.columns[-1] != self.__target_col:
            cols = [c for c in self.__dataset.columns if c != self.__target_col] + [self.__target_col]
            self.__dataset = self.__dataset[cols]
