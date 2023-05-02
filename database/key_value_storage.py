import os
import pandas as pd


class KeyValueStorage():

    def __init__(self, base_path):

        if not base_path:
            raise Exception("base_path is required")

        self.base_path = base_path
        self.storage_path = self.base_path + "/assets/"
        self.file_name = "storage.json"

        # create storage if it doesn't exist
        if not os.path.exists(self.storage_path + self.file_name):
            df = pd.DataFrame(columns=["key", "value"])
            df.to_hdf(self.storage_path + self.file_name, key="df")

        self.storage = pd.read_hdf(self.storage_path + self.file_name, "df")

    def get(self, key):
        _key = key.lower()
        keys = self.storage["key"].tolist()

        if _key in keys:
            return self.storage[self.storage["key"] == key]["value"].tolist()[0]
        return None

    def update(self, key, value):
        _key = key.lower()
        keys = self.storage["key"].tolist()

        if _key in keys:
            self.storage[self.storage["key"] == key]["value"] = value
        else:
            new_row = {"key": key, "value": value}
            new_df = pd.DataFrame([new_row])
            self.storage = pd.concat([self.storage, new_df], ignore_index=True)

            self.storage.to_hdf(self.storage_path + self.file_name, key="df")

    def delete(self, key):
        _key = key.lower()
        keys = self.storage["key"].tolist()

        if _key in keys:
            self.storage[self.storage["key"] == key].drop()
            self.storage.to_hdf(self.storage_path + self.file_name, key="df")

    def delete_all(self):
        self.storage.drop(self.storage.index, inplace=True)
        self.storage.to_hdf(self.storage_path + self.file_name, key="df")

    def show_entries(self):
        pd.set_option('display.max_rows', self.storage.shape[0]+1)
        print(self.storage)
