import logging
import os
import sys
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blueprint.routes import routes_bp
from core.rich_logging import logger as log

from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

app = Flask(__name__)
app.logger.handlers = log.handlers
app.logger.setLevel(logging.INFO)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

app.register_blueprint(routes_bp)


@app.before_request
def before_request():
    log.info(f"Request Received: {request.method} {request.url}")


if __name__ == "__main__":
    time_in = datetime.datetime.now()
    app.run(host='0.0.0.0', port=5000, debug=True)
    time_out = datetime.datetime.now() - time_in
    log.info(f"Application started in {time_out.total_seconds()} seconds.")