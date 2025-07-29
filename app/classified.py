class Classified:
    def __init__(self, features, priors):
        self.__features = features
        self.__priors = priors

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
        for label in self.__priors:
            p = self.__priors[label]
            for feature, value in sample_dict.items():
                p *= self.__features[label][feature].get(value, 0)
            prob[label] = p
        return max(prob, key=prob.get)