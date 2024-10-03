# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:57:49 2023

@author: dm40124
"""

import receptiviti

url = 'http://localhost:4100'
api_key = '1f655ebf74dd4c298cae9c999160339c'
api_secret = 'Yi0eQpsmybLLvzMjUCquA2jSjYp7KR+4kH6wCbFleOsxh3IAd0ArDhJh'

# these 3 change to reflect new inputs and outputs for scoring and results

# If doing single file, use this:
input_file = 'ZS07_HCP_full.csv' # may need to stipulate full path if not in CD
# If doing BULK across multiple files, use this:
# directory = '.' # may need to stipulate full path if not in CD
output_file = 'receptiviti_output2.csv'
text_column = 'Content' #this needs to be the same in all files if doing BULK

# 
receptiviti.request(files = input_file,
    #directory=directory,
                    output=output_file,
                    text_column=text_column,
                    key=api_key,
                    secret=api_secret,
                    url=url,
                    overwrite=True,
                    file_type="csv")