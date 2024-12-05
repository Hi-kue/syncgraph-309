import os
import sys
import datetime
import pickle

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.response import ResponseStatus
from core.orouter_client import OpenRouterClient
from core.rich_logging import logger as log

from flask import request, jsonify, Response
from flask_smorest import Blueprint

from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
MODELS_DIR = os.path.join(ROOT, "server", "models")

routes_bp = Blueprint(
    "routes_bp",
    __name__,
    url_prefix="/api/v1",
    description="Endpoints for the Theft Over Open Data group project.")


def safe_load_models(model_path):
    try:
        with open(model_path, "rb") as file:
            log.info(f"Loading from Model Path: {model_path}")
            model = pickle.load(file)

            if model is None:
                raise Exception("Retrieved model is 'None', something went wrong.")

            return model

    except Exception as e:
        log.error(f"Could not load model: {e}")
        return None


@routes_bp.route("/predict", methods=["GET", "POST"])
def predict() -> tuple[Response, int]:
    log.info("SERVE: /api/v1/predict [GET, POST] route")

    models_dict = {
        "Logistic Regression": "lr_model.pkl",
        "Decision Tree": "dt_model.pkl",
        "Random Forest": "rf_model.pkl"
    }

    try:
        request_json = request.get_json()
        model_name = request.args.get("model_name")

        if not request_json:
            raise ValueError("No JSON was provided in the request.")

        if not model_name:
            raise ValueError("No model name was provided in the request.")

        log.info(f"Received request with json: {request_json} and selected model: {model_name}")

        model_name = models_dict.get(model_name)
        if not model_name:
            raise ValueError(f"Model name '{model_name}' is not supported.")

        model = safe_load_models(os.path.join(MODELS_DIR, model_name))

        query = pd.get_dummies(pd.DataFrame(request_json))

        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(query)

        query = pd.DataFrame(scaled_df)
        confidence = model.predict_proba(query)[0][1]
        prediction = model.predict(query)

        return jsonify({
            "status": ResponseStatus.SUCCESS.value,
            "prediction": {
                "values": f"{', '.join(map(str, prediction))}",
                "confidence": confidence
            },
            "model": f"{model_name}",
            "timestamp": datetime.datetime.now()
        }), 200

    except Exception as e:
        log.error(f"Error processing request: {str(e)}")
        return jsonify({
            "status": ResponseStatus.INTERNAL_SERVER_ERROR.value,
            "message": "An error occurred while processing the request.",
            "data": {
                "error": str(e),
                "model": "Logistic Regression",
                "timestamp": datetime.datetime.now()
            },
            "timestamp": datetime.datetime.now()
        }), 500


@routes_bp.route("/summarize", methods=["POST"])
def summarize() -> tuple[Response, int]:
    log.info("SERVE: /api/v1/summarize POST (summarize route)")

    try:
        request_json = request.get_json()
        log.info(f"Receiving request with json: {request_json}")

        client = OpenRouterClient()
        summary = client.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": ""
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format=None
        )

        if not summary:
            log.error("No summary was provided in the request, something went wrong.")
            return jsonify({
                "status": ResponseStatus.BAD_REQUEST.value,
                "message": "No summary was provided in the request.",
                "data": {
                    "summary": [],
                },
                "timestamp": datetime.datetime.now()
            }), 400

        return jsonify({
            "status": ResponseStatus.SUCCESS.value,
            "message": "An error occurred while processing the request.",
            "data": {
                "summary": summary,
            },
            "timestamp": datetime.datetime.now()
        }), 500

    except Exception as e:
        log.error(f"Error processing the request: {str(e)}")

        return jsonify({
            "status": ResponseStatus.INTERNAL_SERVER_ERROR.value,
            "message": "An error occurred while processing the request.",
            "data": {
                "error": str(e)
            },
            "timestamp": datetime.datetime.now()
        }), 500


@routes_bp.route("/", methods=["GET"])
def health() -> tuple[Response, int]:
    log.info("SERVE: /api/v1/ GET route")
    return jsonify({
        "status": ResponseStatus.SUCCESS.value,
        "message": "Theft Over Open Data Prediction API is running.",
        "data": None,
        "version": "1.0.0",
        "timestamp": datetime.datetime.now()
    }), 200
