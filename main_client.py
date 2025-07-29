import requests

from app.client_api import ClientApi
from app.classified import Classified

response = requests.get("http://server:8000/model")
data = response.json()

features = data["features"]
priors = data["priors"]

model = Classified(features, priors)

controller = ClientApi(model)
app = controller.app
