import torch

from env import API_KEY
from env import BASE_PATH

from database.key_value_storage import KeyValueStorage
from apis.chatgpt_api import ChatgptApi


def fill_db(storage, labels, api):

    for label in labels:

        # also updates the database
        response = api.get_label_response(label)


if __name__ == "__main__":
    storage = KeyValueStorage(BASE_PATH)

    api = ChatgptApi(API_KEY, database=storage)

    yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    labels = list(yolo5.names.values())

    # print('labels: ', len(labels))

    # storage.update("labels", "Test test test test. Tetetetest.")
    # storage.delete_all()

    fill_db(storage, labels, api)

    print(storage.get_all_entries())

    pass
