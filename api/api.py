import uvicorn
from fastapi import FastAPI

from data_loader.csv_data_loader import CsvDataLoader
from naive_bayes_classifier.naive_bayes_classifier import NaiveBayesClassifier

loader = CsvDataLoader("C:\\PyCharm\\Naive_Bayes\\data\\Buy_Computer.csv")
model = NaiveBayesClassifier(loader.df)
model.model_training()

app = FastAPI()

@app.get("/{request}")
async def root(request):
    request = request.split(".")
    s_dic = {}
    for i in range(0, len(request) ,2):
        s_dic[request[i]] = request[i+1]
    return {"answer": model.predict(s_dic)}





if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
