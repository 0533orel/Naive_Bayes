import uvicorn
from fastapi import FastAPI

from csv_data_loader import CsvDataLoader
from naive_bayes_classifier import NaiveBayesClassifier

loader = CsvDataLoader("data.csv")
model = NaiveBayesClassifier(loader.df)
model.model_training()

app = FastAPI()

@app.get("/{name}")
async def root(name):
    name = name.split(".")
    s_dic = {}
    for i in range(0, len(name) ,2):
        s_dic[name[i]] = name[i+1]
    return {"answer": model.predict(s_dic)}





if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
