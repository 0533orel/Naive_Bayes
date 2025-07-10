import copy


class NaiveBayesClassifier:
    def __init__(self, data):
        self._df = data
        self._p = {}
        self._x = {}
        self._n_samples = 0

    def get_dictionaries(self):
        if self._df is None:
            raise ValueError("\nthe DataFrame is empty")

        columns = self._df.columns.to_list()
        label = self._df[columns[-1]].unique().tolist()
        features = {}
        for col in columns[:-1]:
            value = self._df[col].unique().tolist()
            val_dic = {}
            for val in value:
                val_dic[val] = 0
            features[col] = val_dic

        for lbl in label:
            self._x[lbl] = copy.deepcopy(features)
            self._p[lbl] = 0

    def fit(self):
        if not self._x:
            self.get_dictionaries()

        self._n_samples = len(self._df)
        columns = self._df.columns.to_list()

        for _, row in self._df.iterrows():
            row_list = row.values.tolist()
            label = row_list[-1]
            self._p[label] += 1
            for i in range(len(row_list) - 1):
                feature = columns[i]
                value = row_list[i]
                self._x[label][feature][value] += 1

    def model_training(self):
        if not self._x:
            self.fit()

        for label in self._x:
            for feature in self._x[label]:
                k = len(self._x[label][feature])
                for value in self._x[label][feature]:
                    count = self._x[label][feature][value]
                    prob = (count + 1) / (self._p[label] + k)
                    self._x[label][feature][value] = prob
            self._p[label] /= self._n_samples

    def predict(self, sample_dict):
        if not self._x:
            raise ValueError("\nthere is no data in dictionary")

        prob = {}
        for label in self._p:
            p = self._p[label]
            for feature, value in sample_dict.items():
                p *= self._x[label][feature].get(value, 0)
            prob[label] = p
        return max(prob, key=prob.get)
