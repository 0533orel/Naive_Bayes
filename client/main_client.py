import requests

from api.client_api import ClientApi
from model.classified import Classified

response = requests.get("http://server:8000/model")
data = response.json()

features = data["features"]
priors = data["priors"]

model = Classified(features, priors)

controller = ClientApi(model)
app = controller.app
