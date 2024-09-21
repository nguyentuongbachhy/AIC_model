import json, os


SOURCE_PATH = "D:/AIC/model/assets/results"

def sort_dict(dict_path):
    with open(dict_path, 'r') as f:
        file = json.load(f)

    file = {
        key: file[key] for key in sorted(file.keys(), key=lambda key: int(key))
    }

    with open(dict_path, 'w') as f:
        json.dump(file, f, indent=4)

def sort_all_dicts_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = f'{folder_path}/{filename}'
        sort_dict(file_path)
        print(f"Sorted all items in {filename} successfully!")


sort_all_dicts_in_folder(SOURCE_PATH)
