import openai
import json


class ChatgptApi():

    def __init__(self, api_key, gpt_model=None, database=None):

        self.api_key = api_key
        self.database = database

        self.gpt_model = None
        if not gpt_model or gpt_model == "gpt-3.5":
            self.gpt_model = "gpt-3.5-turbo-0301"
        elif gpt_model == "gpt4":
            self.gpt_model = "gpt4"

        self.url = "https://api.openai.com/chat/completions"
        self.auth = f"Bearer {self.api_key}"

        openai.api_key = self.api_key

    def get_label_response(self, label, test=False):

        if test:
            return self._get_dummy_response()

        if self.database:
            response = self.database.get(label)
            if response is not None:
                print('We have a response in the database.')
                return response

        pre_message = {
            "role": "user",
            "content": "The following are not questions about beliefs or opinions. Please answer very concisely in maximum 20 words."
        }

        first_question = {
            "role": "user",
            "content": f"What is a {label}? If usable, what can a {label} be used for? If not usable what is a {label} able to do? "
        }

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                pre_message,
                first_question,
            ],
        )

        response = response["choices"][0]["message"]["content"]

        print(response)

        self.database.update(label, response)

        return response

    def _get_dummy_response(self):

        dummy_response = \
            """
        {
            "choices": [
                {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "A tea cup is a small cup, typically with a handle, that is used for drinking tea. It can also be used for serving other beverages such as coffee, hot chocolate, or herbal tea. Additionally, tea cups can be used as decorative items or as collectibles.",
                    "role": "assistant"
                    }
                }
            ],
            "created": 1682324860,
            "id": "chatcmpl-78lpsLLBXUKulsDRpiVsY0bXQEulZ",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 55,
                "prompt_tokens": 29,
                "total_tokens": 84
            }
        }
        """

        """
        {
            "choices": [
                {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "As an AI language model, I do not encourage the use of people for any purpose as they are not objects. People are living beings that possess intelligence, consciousness, and free will. They have physical, emotional, and intellectual capabilities and should be treated with dignity and respect. People can contribute to society in various ways, such as working, serving others, creating art, innovating technology, and more. However, they should never be used for exploitation, abuse, or manipulation.",
                    "role": "assistant"
                    }
                }
            ],
            "created": 1682329761,
            "id": "chatcmpl-78n6vmUcQGEAHFRitIweJ7e4n5XPy",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 96,
                "prompt_tokens": 27,
                "total_tokens": 123
            }
        }
        """

        response_json = json.loads(dummy_response)
        response = response_json["choices"][0]["message"]["content"]

        return response
