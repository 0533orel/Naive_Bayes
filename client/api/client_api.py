from fastapi import FastAPI

class ClientApi:
    def __init__(self, model):
        self.app = FastAPI()
        self.model = model

        @self.app.get("/{request}")
        async def root(request):
            request = request.split(".")
            s_dic = {}
            for i in range(0, len(request) ,2):
                s_dic[request[i]] = request[i+1]
            return {"answer": self.model.predict(s_dic)}
