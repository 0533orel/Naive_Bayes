import uvicorn
from data_loader.loader import LoadData
from data_loader.cleaner import Cleaner
from model.naive_bayes_classifier import NaiveBayesClassifier
from api.api_controller import ApiController


data = LoadData("data/Buy_Computer.csv")
data = Cleaner(data.dataset, data.target_col)
model = NaiveBayesClassifier(data.dataset)
model.model_training()
app = ApiController(model)


if __name__ == "__main__":
    uvicorn.run(app.app, host="127.0.0.1", port=8000)