import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import OPENROUTER_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_URL
from core.rich_logging import logger as log

from openai import OpenAI, OpenAIError

DEFAULT_SYSTEM_PROMPT = """
You are a skilled data scientist tasked with evaluating the performance of predictive models applied
to datasets from the Toronto Police Portal. Your goal is to analyze the model outputs and assess 
their efficiency in predicting theft types based on specific features. Follow the instructions 
below to provide a concise and thorough analysis.

The models are specified in the JSON data under the `model_name` field, with the following options:
1. `LogisticRegression`
2. `DecisionTreeClassifier`
3. `RandomForestClassifier`

SMOTE (smote):
Features used: `LOCATION_TYPE`, `PREMISES_TYPE`
Example JSON:
   {
       "model": "lr_model.pkl",
       "prediction": {
           "confidence": 0.8198601086183642,
           "values": "1, 0"
       },
       "status": 200,
       "timestamp": "Tue, 10 Dec 2024 23:41:16 GMT"
   }
   
---
SMOTENC (smotenc):
Features used: LOCATION_TYPE, PREMISES_TYPE, HOOD_158, LONG_WGS84, LAT_WGS84, OCC_HOUR, REPORT_HOUR
Example JSON:
    {
        "model": "lr_model_smotenc.pkl",
        "prediction": {
            "confidence": 0.8150654624056394,
            "values": "1"
        },
        "status": 200,
        "timestamp": "Wed, 11 Dec 2024 01:35:22 GMT"
    }
    
---
Prediction Details

The prediction.values field contains a comma-separated string of predicted outcomes:
0 indicates Theft From Motor Vehicle Over $5000.
1 indicates Theft Over $5000.

The model's objective is to classify whether a given location is more likely to experience 
a Motor Vehicle Theft Over $5000 or a Theft Over $5000.

---
Required Output Format

Your output must include the following sections and should not 
from the format provided below:

1. Prediction Interpretation:
    - Prediction Values: Describe the predicted values and their meaning.
    - Rationale: Explain why the model might have made this classification based on the features.
2. Confidence Analysis:
    - Score: Provide the confidence score.
    - Analysis: Discuss what this score indicates about the model's certainty.
3. Insights:
    - Operational Insights: Highlight key takeaways or recommendations for analysts based on the prediction.
    - Structure: Organized into clearly labeled sections for easy understanding and navigation.
    - Output Format: Added a required output format to ensure consistency and clarity in responses.
    
To ensure accuracy and completeness, adhere to the following guidelines:
- Do not include titles but include headers for each section bolded in the format of:
    - ### Section Title
    - #### Sub-Section Title
- Do not separate sections but when necessary, use a new line.
- Do not include any additional text or explanations beyond the required sections.
- Do not include any unnecessary or redundant information.
- Include styling within the paragraphs to enhance readability.
- Ensure that the output is concise and to the point.
"""


class OpenRouterClient:
    def __init__(self):
        self.__api_url = os.getenv("OPENROUTER_API_BASE_URL")
        self.__api_key = os.getenv("OPENROUTER_API_KEY")
        self.__client = OpenAI(
            base_url=OPENROUTER_API_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers={
                "Authorization": f"Bearer {self.__api_key}",
                "Content-Type": "application/json"
            },
        )

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
            "response_format": response_format
        }

        # NOTE: Numerical value checks for <= 0
        check_dict = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]

        for item in check_dict:
            match item:
                case value if isinstance(value, int) or isinstance(value, float):
                    if value < 0:
                        log.error(f"Invalid value for {item} was provided: {value}")
                        raise ValueError(f"Invalid value for {item} was provided: {value}")

                    if value == 0:
                        log.info(f"Value for {item} is 0, ignoring it in the request.")
                        kwargs.pop(item)
                case _:
                    pass

        try:
            response = self.__client.chat.completions.create(**kwargs)

        except OpenAIError as e:
            log.error(f"Error creating chat completion: {str(e)}")
            raise e

        except Exception as e:
            log.error(f"Other error occurred while creating chat completion: {str(e)}")
            raise e

        log.info(f"Response from OpenRouter API: {response}")
        return response.choices[0].message.content
