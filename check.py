import json, os

RESULT_PATH = 'D:/AIC/model/assets/results'
STORAGE_PATH = 'D:/AIC/model/assets/storage'

for filename in os.listdir(STORAGE_PATH):
    result_file = f'{RESULT_PATH}/{filename}'
    storage_file = f'{STORAGE_PATH}/{filename}'

    result_file_size = os.path.getsize(result_file)
    storage_file_size = os.path.getsize(storage_file)

    if storage_file_size > result_file_size:
        print(f'Error in {filename}')
    
    else:
        with open(result_file, 'r') as file:
            result = json.load(file)

        with open(storage_file, 'r') as file:
            storage = json.load(file)

        if(len(storage) != len(result)):
            print(f'Error in {filename}')
    
