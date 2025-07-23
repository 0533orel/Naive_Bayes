from api.api_controller import ApiController
from data_loader.cleaner import Cleaner
from data_loader.loader import LoadData
from model.naive_bayes_classifier import NaiveBayesClassifier

data = LoadData("data/Buy_Computer.csv")
data = Cleaner(data.dataset, data.target_col)
model = NaiveBayesClassifier(data.dataset)
model.model_training()

controller = ApiController(model)
app = controller.app
