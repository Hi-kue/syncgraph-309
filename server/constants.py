"""
file: constants.py
authors: []
dated: 2024-11-14
description:
    - Constant .env variables used throughout the project in a central location for easy access.
    - This file is not intended to be edited unless you are adding new constants.
    - If you need to add a new constant, add it to the .env file and import it here.
"""

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")
