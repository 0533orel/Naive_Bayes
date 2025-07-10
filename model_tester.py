from naive_bayes_classifier import NaiveBayesClassifier


class ModelTester:
    def __init__(self, data):
        try:
            self.__df = data.sample(frac=1).reset_index(drop=True)
            self.__size_df = int(len(self.__df) * 0.7)
            self.__train_df = self.__df[:self.__size_df]
            self.__test_df = self.__df[self.__size_df:]
            self._model = None
        except Exception as e:
            print(f"\nError: {e}")

    def train_model(self):
        try:
            self._model = NaiveBayesClassifier(self.__train_df)
            self._model.model_training()
        except Exception as e:
            print(f"\nError: {e}")

    def test_model(self):
        if self._model is None:
            self.train_model()

        test_df = self.__test_df.iloc[:, :-1]
        test_len_rows = len(self.__test_df)
        successful_answer = 0
        for row in range(len(test_df)):
            test_dic = test_df.iloc[row].to_dict()
            answer = self._model.predict(test_dic)
            if answer == self.__test_df.iloc[row, -1]:
                successful_answer += 1
        return f"\nsuccessful_answer {successful_answer}/{test_len_rows} -> {int((successful_answer * 100) / test_len_rows)}% success"

