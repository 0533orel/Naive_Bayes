import copy


class NaiveBayesClassifier:
    def __init__(self, data):
        self.df = data
        self.p = {}
        self.x = {}
        self.n_samples = 0

    def get_dictionaries(self):
        if self.df is None:
            raise ValueError("the DataFrame is empty")

        columns = self.df.columns.to_list()
        label = self.df[columns[-1]].unique().tolist()
        features = {}
        for col in columns[:-1]:
            value = self.df[col].unique().tolist()
            val_dic = {}
            for val in value:
                val_dic[val] = 0
            features[col] = val_dic

        for lbl in label:
            self.x[lbl] = copy.deepcopy(features)
            self.p[lbl] = 0

    def fit(self):
        self.get_dictionaries()
        self.n_samples = len(self.df)
        columns = self.df.columns.to_list()

        for _, row in self.df.iterrows():
            row_list = row.values.tolist()
            label = row_list[-1]
            self.p[label] += 1
            for i in range(len(row_list) - 1):
                feature = columns[i]
                value = row_list[i]
                self.x[label][feature][value] += 1

    def model_training(self):
        if not self.x:
            raise ValueError("there is no data in dictionary")

        for label in self.x:
            for feature in self.x[label]:
                k = len(self.x[label][feature])
                for value in self.x[label][feature]:
                    count = self.x[label][feature][value]
                    prob = (count + 1) / (self.p[label] + k)
                    self.x[label][feature][value] = prob
            self.p[label] /= self.n_samples

    def predict(self, sample_dict):
        if not self.x:
            raise ValueError("there is no data in dictionary")

        prob = {}
        for label in self.p:
            p = self.p[label]
            for feature, value in sample_dict.items():
                p *= self.x[label][feature].get(value, 0)
            prob[label] = p
        return max(prob, key=prob.get)
