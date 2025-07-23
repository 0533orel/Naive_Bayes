import copy


class NaiveBayesClassifier:
    """
    A Naive Bayes classifier for categorical data using Laplace smoothing.

    Attributes:
        __dataset (pd.DataFrame): The dataset used for training.
        __probs (dict): Prior probabilities for each class label.
        __features (dict): Conditional probabilities for each feature value given a label.
        __num_samples (int): Total number of training samples.
    """

    def __init__(self, data):
        """
        Initializes the classifier with the given dataset.

        Args:
            data (pd.DataFrame): The dataset including features and target label.
        """
        self.__dataset = data
        self.__probs = {}
        self.__features = {}
        self.__num_samples = 0

    @property
    def dataset(self):
        """
        Returns the cleaned DataFrame.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        return self.__dataset

    @property
    def features(self):
        """
        Returns the conditional probabilities dictionary.

        Returns:
            dict: A nested dictionary of conditional probabilities.
        """
        return self.__features

    def get_dictionaries(self):
        """
        Initializes internal structures for counting feature occurrences per label.
        Raises an error if the dataset is empty.
        """
        if self.__dataset is None:
            raise ValueError("\nthe DataFrame is empty")

        columns = self.__dataset.columns.to_list()
        label = self.__dataset[columns[-1]].unique().tolist()
        features = {}
        for col in columns[:-1]:
            value = self.__dataset[col].unique().tolist()
            val_dic = {}
            for val in value:
                val_dic[val] = 0
            features[col] = val_dic

        for lbl in label:
            self.__features[lbl] = copy.deepcopy(features)
            self.__probs[lbl] = 0

    def fit(self):
        """
        Counts occurrences of feature values per label to prepare for training.
        Automatically calls `get_dictionaries()` if structures are not initialized.
        """
        if not self.__features:
            self.get_dictionaries()

        self.__num_samples = len(self.__dataset)
        columns = self.__dataset.columns.to_list()

        for _, row in self.__dataset.iterrows():
            row_list = row.values.tolist()
            label = row_list[-1]
            self.__probs[label] += 1
            for i in range(len(row_list) - 1):
                feature = columns[i]
                value = row_list[i]
                self.__features[label][feature][value] += 1

    def model_training(self):
        """
        Applies Laplace smoothing and calculates conditional probabilities.
        Converts frequency counts into probabilities.
        """
        if not self.__features:
            self.fit()

        for label in self.__features:
            for feature in self.__features[label]:
                k = len(self.__features[label][feature])
                for value in self.__features[label][feature]:
                    count = self.__features[label][feature][value]
                    prob = (count + 1) / (self.__probs[label] + k)
                    self.__features[label][feature][value] = prob
            self.__probs[label] /= self.__num_samples

    def predict(self, sample_dict):
        """
        Predicts the label for a given sample based on the trained model.

        Args:
            sample_dict (dict): A dictionary mapping feature names to their values.

        Returns:
            str: The predicted label with the highest probability.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.__features:
            raise ValueError("\nthere is no data in dictionary")

        prob = {}
        for label in self.__probs:
            p = self.__probs[label]
            for feature, value in sample_dict.items():
                p *= self.__features[label][feature].get(value, 0)
            prob[label] = p
        return max(prob, key=prob.get)
