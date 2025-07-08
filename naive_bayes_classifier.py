import copy


class NaiveBayesClassifier:
    def __init__(self, data):
        self.df = data
        self.dic = {}

    def get_dictionary_of_values(self):
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
            self.dic[lbl] = copy.deepcopy(features)



