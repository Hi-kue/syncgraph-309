import os
import sys
from typing import List, Dict, Any, Optional

import openai

class OpenRouterClient:
    def __init__(self):
        self.__client = OpenAI()
        self.__api_url = os.getenv("OPENROUTER_API_BASE_URL")
        self.__api_key = os.getenv("OPENROUTER_API_KEY")
        self.__model = "google/gemini-flash-1.5-exp"

    def create_chat_completion(self,
                               messages: List[Dict[str, str]],
                               temperature: float = 1,
                               max_tokens: int = 256,
                               top_p: float = 1,
                               frequency_penalty: float = 0,
                               presence_penalty: float = 0,
                               response_format: Dict = None) -> Any:
        kwargs = {
            "model": self.__model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "response_format": response_format
        }