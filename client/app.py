import logging
import os
import sys
import datetime

import constants

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blueprint.routes import routes_bp
from core.rich_logging import logger as log
from constants import FLASK_APP, FLASK_ENV, FLASK_RUN_PORT

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
app.logger.handlers = log.handlers
app.logger.setLevel(logging.INFO)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["FLASK_APP"] = FLASK_APP
app.config["FLASK_ENV"] = FLASK_ENV
app.config["FLASK_RUN_PORT"] = FLASK_RUN_PORT

app.register_blueprint(routes_bp)


@app.before_request
def before_request():
    log.info(f"Request Received: {request.method} {request.url}")


if __name__ == "__main__":
    time_in = datetime.datetime.now()
    app.run(host='0.0.0.0', port=5000, debug=True)
    time_out = datetime.datetime.now() - time_in
    log.info(f"Application started in {time_out.total_seconds()} seconds.")