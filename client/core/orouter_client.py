import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import OPENROUTER_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_URL
from core.rich_logging import logger as log

from openai import OpenAI, OpenAIError

DEFAULT_SYSTEM_PROMPT = """
<WRITE_SYSTEM_PROMPT_HERE>
"""


class OpenRouterClient:
    def __init__(self):
        self.__client = OpenAI(
            base_url=OPENROUTER_API_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.__api_url = os.getenv("OPENROUTER_API_BASE_URL")
        self.__api_key = os.getenv("OPENROUTER_API_KEY")

    @property
    def client(self) -> OpenAI:
        return self.__client

    @client.setter
    def client(self, client: OpenAI) -> None:
        self.__client = client

    @property
    def api_url(self) -> str:
        return self.__api_url

    @api_url.setter
    def api_url(self, url: str) -> None:
        self.__api_url = url

    def create_chat_completion(self,
                               messages: List[Dict[str, str]],
                               temperature: float = 1,
                               max_tokens: int = 256,
                               top_p: float = 1,
                               frequency_penalty: float = 0,
                               presence_penalty: float = 0,
                               response_format: Dict = None) -> Any:
        kwargs = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT
                },
                *messages
            ],
            "temperature": temperature or 1,
            "max_tokens": max_tokens,
            "top_p": top_p or 1,
            "frequency_penalty": frequency_penalty or 0,
            "presence_penalty": presence_penalty or 0,
            "response_format": response_format or None
        }

        try:
            response = self.__client.chat.completions.create(**kwargs)

        except OpenAIError as e:
            log.error(f"Error creating chat completion: {str(e)}")
            raise e

        except Exception as e:
            log.error(f"Other error occured while creating chat completion: {str(e)}")
            raise e

        return response.choices[0].message.content
