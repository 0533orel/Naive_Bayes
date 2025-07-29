from app.cleaner import Cleaner
from app.loader import LoadData
from app.naive_bayes_classifier import NaiveBayesClassifier
from app.server_api import ServerApi

data = LoadData("data/Buy_Computer.csv")
data = Cleaner(data.dataset, data.target_col)
model = NaiveBayesClassifier(data.dataset)
model.model_training()

controller = ServerApi(model)
app = controller.app
