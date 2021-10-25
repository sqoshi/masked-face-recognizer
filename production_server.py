"""
Installing:
    - pip install fastapi
    - pip install "uvicorn[standard]"

Running:
     - python3 -m uvicorn production_server:app --reload

"""
import json
import os

from fastapi import FastAPI

from src.settings import output

app = FastAPI()


def handle_dir(path):
    dictionary = {"files": [], "directories": []}
    for entity in os.listdir(path):
        abs_path = str(path / entity)
        if os.path.isdir(abs_path):
            dictionary["directories"] += [entity]
        elif os.path.isfile(abs_path):
            dictionary["files"] += [entity]
    return dictionary


def handle_file(path):
    dictionary = {"filename": path.name}
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


@app.get("/output/{dataset}/{analysis_group}")
def list_analysis(dataset, analysis_group):
    return build_dict(output / dataset / analysis_group)


@app.get("/output/{dataset}/{analysis_group}/{research_id}/analysis_config")
def show_analysis_config(dataset, analysis_group, research_id):
    return build_dict(output / dataset / analysis_group / research_id / "analysis_config.json")


@app.get("/output/{dataset}/{analysis_group}/{research_id}/model_config")
def show_model_config(dataset, analysis_group, research_id):
    return build_dict(output / dataset / analysis_group / research_id / "model_config.json")


@app.get("/output/{dataset}/{analysis_group}/{research_id}/statistics")
def show_model_config(dataset, analysis_group, research_id):
    return build_dict(output / dataset / analysis_group / research_id / "results.csv")
