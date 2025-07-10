from naive_bayes_classifier import NaiveBayesClassifier


class ModelTester:
    def __init__(self, data):
        self.df = data.sample(frac=1).reset_index(drop=True)
        self.size_df = int(len(self.df) * 0.7)
        self.train_df = self.df[:self.size_df]
        self.test_df = self.df[self.size_df:]
        self.model = None

    def train_model(self):
        self.model = NaiveBayesClassifier(self.train_df)
        self.model.get_dictionaries()
        self.model.fit()
        self.model.model_training()

    def test_model(self):
        if self.model is None:
            self.train_model()
        test_df = self.test_df.iloc[:, :-1]
        test_len_rows = len(self.test_df)
        successful_answer = 0
        for row in range(len(test_df)):
            test_dic = test_df.iloc[row].to_dict()
            answer = self.model.predict(test_dic)
            if answer == self.test_df.iloc[row, -1]:
                successful_answer += 1
        print(f"successful_answer {successful_answer}/{test_len_rows} -> {int((successful_answer * 100) / test_len_rows)}% success")

