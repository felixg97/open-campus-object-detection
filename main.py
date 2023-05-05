
import torch

from env import API_KEY
from env import BASE_PATH
from database.key_value_storage import KeyValueStorage
from apis.chatgpt_api import ChatgptApi
from object_detection_app import ObjectDetectionApp


if __name__ == "__main__":

    storage = KeyValueStorage(BASE_PATH)

    api = ChatgptApi(API_KEY, database=storage)

    yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    app = ObjectDetectionApp(model=yolo5, api=api, base_path=BASE_PATH)

    if False:
        storage.delete_all()
        exit()

    if False:
        storage.show_entries()
        exit()

    if True:
        app.start_capturing(cam=0)

    pass
