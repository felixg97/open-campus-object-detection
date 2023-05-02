import openai
import json


class ChatgptApi():

    def __init__(self, api_key, gpt_model=None, keyvalue_storage=None, mode="default"):

        self.api_key = api_key
        self.keyvalue_storage = keyvalue_storage
        self.mode = mode  # default, delete, test

        self.gpt_model = None
        if not gpt_model or gpt_model == "gpt-3.5":
            self.gpt_model = "gpt-3.5-turbo-0301"
        elif gpt_model == "gpt4":
            self.gpt_model = "gpt4"

        self.url = "https://api.openai.com/chat/completions"
        self.auth = f"Bearer {self.api_key}"

        openai.api_key = self.api_key

    def get_label_response(self, label):

        if self.mode == "test":
            return self._get_dummy_reponse()

        if self.keyvalue_storage:
            response = self.keyvalue_storage.get(label)
            if response:
                return response

        pre_message = {
            "role": "user",
            "content": "This following are not questions about beliefs or opinions. Please answer very concisely."
        }

        first_question = {
            "role": "user",
            "content": f"What is a {label}? "
        }

        response1 = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                pre_message,
                first_question
            ],
        )
        second_question = {
            "role": "user",
            "content": f"What can a {label} be used for or what is a {label} able to do?"
        }

        response2 = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                pre_message,
                second_question
            ],
        )

        response1 = response1["choices"][0]["message"]["content"]
        response2 = response2["choices"][0]["message"]["content"]

        response = response1 + " " + response2

        self.keyvalue_storage.update(label, response)

        return response

    def _get_dummy_reponse(self):

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
