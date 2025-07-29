from fastapi import FastAPI


class ServerApi:
    def __init__(self, model):
        self.app = FastAPI()
        self.model = model

        @self.app.get("/model")
        def get_model():
            return {
                "features": model.features,
                "priors": model.probs
            }