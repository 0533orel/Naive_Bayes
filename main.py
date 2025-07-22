from controller.controller import ControllerApi
from data_loader.csv_data_loader import CsvDataLoader
from naive_bayes_classifier.naive_bayes_classifier import NaiveBayesClassifier

loader = CsvDataLoader("data/Buy_Computer.csv")
model = NaiveBayesClassifier(loader.df)
model.model_training()

controller = ControllerApi(model)
app = controller.app
