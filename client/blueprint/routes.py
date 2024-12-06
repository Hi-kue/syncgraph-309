import os
import sys
import datetime
from enum import Enum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rich_logging import logger as log
from core.model_funcs import safe_load_models

import pandas as pd
from flask import request, jsonify, Response
from flask_smorest import Blueprint

from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
MODELS_DIR = os.path.join(ROOT, "server", "models")


class ResponseStatus(Enum):
    SUCCESS = 200
    NOT_FOUND = 404
    BAD_REQUEST = 400
    INTERNAL_SERVER_ERROR = 500
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    CONFLICT = 409


routes_bp = Blueprint(
    "routes_bp",
    __name__,
    url_prefix="/api/v1",
    description="Endpoints for the Theft Over Open Data group project.")


@routes_bp.route("/predict", methods=["GET", "POST"])
def predict() -> tuple[Response, int]:
    """
    :endpoint: /api/v1/predict
    :methods: GET, POST
    :description:
        - This is the old endpoint for model predictions.
        - Models that are trained without SMOTENC are the following:
            - Logistic Regression: lr_model.pkl
            - Decision Tree: dt_model.pkl
            - Random Forest: rf_model.pkl

    :raises:
        - ValueError is no JSON was provided in the request.
        - ValueError if no model name was provided in the request.
        - ValueError if the model name is not supported.

    :returns:
        - status code 200 if successful (200 OK) - Response 200 OK
        - Status code 400 if it's a bad request (400 Bad Request)
    """
    log.info("SERVE: /api/v1/predict [GET, POST] route")

    models_dict = {
        "LogisticRegression": "lr_model.pkl",
        "DecisionTreeClassifier": "dt_model.pkl",
        "RandomForestClassifier": "rf_model.pkl"
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
        }), 400


@routes_bp.route("/predict/smotenc", methods=["GET", "POST"])
def predict_smotenc() -> tuple[Response, int]:
    """
    :endpoint: /api/v1/predict/smotenc
    :methods: GET, POST
    :description:
        - This is the new endpoint for model predictions with SMOTENC.
        - Models that are trained with SMOTENC are the following:
            - Logistic Regression: lr_model_smotenc.pkl
            - Decision Tree: dt_model_smotenc.pkl
            - Random Forest: rf_model_smotenc.pkl

    :raises:
        - ValueError is no JSON was provided in the request.
        - ValueError if no model name was provided in the request.
        - ValueError if the model name is not supported.

    :returns:
        - status code 200 if successful (200 OK) - Response 200 OK
        - Status code 400 if it's a bad request (400 Bad Request)
    """
    log.info("SERVE: /api/v1/predict/smotenc [GET, POST] route")

    models_dict = {
        "LogisticRegression": "lr_model_smotenc.pkl",
        "DecisionTreeClassifier": "dt_model_smotenc.pkl",
        "RandomForestClassifier": "rf_model_smotenc.pkl"
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
            "status": ResponseStatus.BAD_REQUEST.value,
            "message": "An error occurred while processing the request.",
            "data": {
                "error": str(e),
                "model": "Logistic Regression",
                "timestamp": datetime.datetime.now()
            },
            "timestamp": datetime.datetime.now()
        }), 400


@routes_bp.route("/", methods=["GET"])
def health() -> tuple[Response, int]:
    """
    :endpoint: /api/v1/
    :methods: GET
    :description:
        - This is the general endpoint for the API, used to check if the API is running.
        - This endpoint will return a 200 OK status code if the API is running.

    :return:
        - status code 200 if successful (200 OK) - Response 200 OK
    """
    log.info("SERVE: /api/v1/ GET route")
    return jsonify({
        "status": ResponseStatus.SUCCESS.value,
        "message": "Theft Over Open Data Prediction API is running.",
        "data": None,
        "version": "1.0.0",
        "timestamp": datetime.datetime.now()
    }), 200
