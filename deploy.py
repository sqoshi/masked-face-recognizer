"""
Installing:
    - pip install fastapi "uvicorn[standard]"

Running:
     - python3 -m uvicorn production_server:app --reload

"""
import json
import os

import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from src.settings import output

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def handle_dir(path):
    dictionary = {"type": "directory", "files": [], "directories": []}
    for entity in os.listdir(path):
        abs_path = str(path / entity)
        if os.path.isdir(abs_path):
            dictionary["directories"] += [entity]
        elif os.path.isfile(abs_path):
            dictionary["files"] += [entity]
    return dictionary


def handle_file(path):
    dictionary = {"type": "file", "filename": path.name}
    with open(path, "r") as fr:
        if str(path).endswith(".json"):
            dictionary["content"] = json.load(fr)
        else:
            dictionary["content"] = fr.read()

    return dictionary


def build_dict(path):
    if os.path.isfile(path):
        return handle_file(path)
    return handle_dir(path)


@app.get("/")
def read_root():
    return {"info": "This api allow to share results of researches."}


@app.get("/output")
def list_datasets():
    return build_dict(output)


@app.get("/output/{dataset}")
def list_researches(dataset):
    return build_dict(output / dataset)


@app.get("/output/{dataset}/{research_group}")
def list_analysis(dataset, research_group):
    return build_dict(output / dataset / research_group)


@app.get("/output/{dataset}/{research_group}/{analysis_id}/analysis_config")
def show_analysis_config(dataset, research_group, analysis_id):
    return build_dict(output / dataset / research_group / analysis_id / "analysis_config.json")


@app.get("/output/{dataset}/{research_group}/{analysis_id}/model_config")
def show_model_config(dataset, research_group, analysis_id):
    return build_dict(output / dataset / research_group / analysis_id / "model_config.json")


@app.get("/output/{dataset}/{research_group}/{analysis_id}/statistics")
def show_model_config(dataset, research_group, analysis_id):
    return build_dict(output / dataset / research_group / analysis_id / "results.csv")


if __name__ == '__main__':
    uvicorn.run(app, port=8668, host='127.0.0.1')
