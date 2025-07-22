from api.api_controller import ApiController
from data_loader.cleaner import Cleaner
from data_loader.loader import LoadData
from model.naive_bayes_classifier import NaiveBayesClassifier

loader = LoadData("data/Buy_Computer.csv")
cleaner = Cleaner(loader.dataset, loader.target_col)
model = NaiveBayesClassifier(cleaner.dataset)
model.model_training()

controller = ApiController(model)
app = controller.app
