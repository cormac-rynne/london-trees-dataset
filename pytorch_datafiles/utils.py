import json
import os

def read_json_filelist(json_filepath):

    data_root_directory, filename = os.path.split(json_filepath)

    with open(json_filepath,'r') as f:
        file_lst = json.load(f)

    all_file_lst = []

    for file in file_lst:
        img_filepath = os.path.join(data_root_directory, *file[0])
        den_filepath = os.path.join(data_root_directory, *file[1])
        all_file_lst.append((img_filepath, den_filepath))

    return all_file_lst