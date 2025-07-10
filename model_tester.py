from naive_bayes_classifier import NaiveBayesClassifier

class ModelTester:
    """
    A helper class to split data, train a Naive Bayes classifier, and test its accuracy.

    Attributes:
        __df (pd.DataFrame): The full shuffled dataset.
        __train_df (pd.DataFrame): The training portion of the dataset (70%).
        __test_df (pd.DataFrame): The testing portion of the dataset (30%).
        __model (NaiveBayesClassifier): The trained Naive Bayes model.
    """

    def __init__(self, data):
        """
        Splits the provided dataset into training and testing sets.

        Args:
            data (pd.DataFrame): The full dataset to be used.
        """
        try:
            self.__df = data.sample(frac=1).reset_index(drop=True)
            self.__size_df = int(len(self.__df) * 0.7)
            self.__train_df = self.__df[:self.__size_df]
            self.__test_df = self.__df[self.__size_df:]
            self.__model = None
        except Exception as e:
            print(f"\nError: {e}")

    @property
    def model(self):
        """
        Returns the trained model instance.

        Returns:
            NaiveBayesClassifier: The trained classifier.
        """
        return self.__model

    def train_model(self):
        """
        Initializes and trains the Naive Bayes classifier using the training dataset.
        """
        try:
            self.__model = NaiveBayesClassifier(self.__train_df)
            self.__model.model_training()
        except Exception as e:
            print(f"\nError: {e}")

    def test_model(self):
        """
        Evaluates the accuracy of the trained model on the test dataset.

        Returns:
            str: A summary string indicating number of correct predictions and success percentage.
        """
        if self.__model is None:
            self.train_model()

        test_df = self.__test_df.iloc[:, :-1]
        test_len_rows = len(self.__test_df)
        successful_answer = 0
        for row in range(len(test_df)):
            test_dic = test_df.iloc[row].to_dict()
            answer = self.__model.predict(test_dic)
            if answer == self.__test_df.iloc[row, -1]:
                successful_answer += 1
        return f"\nsuccessful_answer {successful_answer}/{test_len_rows} -> {int((successful_answer * 100) / test_len_rows)}% success"
