import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

MODEL_PROMPT = """"
You are a Scientific Data Analyst <add more details about your role here>.
<add information about the task you need to complete here>.
<add more information about the details of the task here>.
"""
