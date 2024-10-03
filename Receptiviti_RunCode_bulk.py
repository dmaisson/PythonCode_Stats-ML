# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:57:49 2023

@author: dm40124
"""

import receptiviti
import os

url = 'http://localhost:4100'
api_key = '1f655ebf74dd4c298cae9c999160339c'
api_secret = 'Yi0eQpsmybLLvzMjUCquA2jSjYp7KR+4kH6wCbFleOsxh3IAd0ArDhJh'
id_column = 'Speaker'
text_column = 'Content'

path = os.getcwd()
dir_list = os.listdir(path)
iA = 0
while iA < len(dir_list):
    x = dir_list[iA]
    y = x.split(".")
    if len(y) >= 2:
        if (y[-1] == 'csv'):
            iA += 1
        else:
            dir_list.pop(iA)
            iA = 0
    else:
        dir_list.pop(iA)
        iA = 0

iA = 0
while iA < len(dir_list):
    input_file = dir_list[iA]
    x = input_file.split("_")
    output_file = x[0] + '_output.csv'
    receptiviti.request(files = input_file, #directory=directory,
                        output=output_file,
                        text_column=text_column,
                        id_column=id_column,
                        key=api_key,
                        secret=api_secret,
                        url=url,
                        cores=1,
                        overwrite=True,
                        encoding='utf-8',
                        file_type="csv")
    iA += 1
    

# import pandas
# results = pandas.read_csv("Cleaned_output.csv", index_col="id")
# ids = ["Cleaned_Data_Receptiviti_mod.csv" + str(i + 1) for i in range(len(results))]
# results_reordered = results.sort_values(by="id", key=lambda x: [int(id[32:]) for id in x])
# results_reordered.to_csv("Cleaned_output_corrected.csv")
