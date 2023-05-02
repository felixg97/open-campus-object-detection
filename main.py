
import torch

from PIL import Image
from env import API_KEY
from env import BASE_PATH
from database.key_value_storage import KeyValueStorage
from apis.chatgpt import ChatgptApi
from window_capture import WindowCapture


if __name__ == "__main__":

    storage = KeyValueStorage(BASE_PATH)

    api = ChatgptApi(API_KEY, keyvalue_storage=storage)

    yolo5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    window_capture = WindowCapture(model=yolo5, api=api, base_path=BASE_PATH)

    if False:
        storage.delete_all()
        exit()

    if False:
        storage.show_entries()
        exit()

    if True:
        window_capture.start_capturing(cam=0)

    pass
