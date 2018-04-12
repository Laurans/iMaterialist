# coding utf-8

import json
import urllib.request
import tqdm
import pickle

with open('data/raw/fgvc4_iMat.test.image.json') as data_file:
    data = json.load(data_file)

# Download test image
data = data['images']
folder_path = 'data/test'

# errors = []
# print(len(data))
# for dict_item in tqdm.tqdm(data, desc='Get test images'):
#     filename = '{}/{}.jpg'.format(folder_path, dict_item['imageId'])
#
#     for url in dict_item['url']:
#         try :
#             request = urllib.request.urlopen(url, timeout=50)
#         except:
#             errors.append(dict_item)
#             continue
#         try:
#             urllib.request.urlretrieve(url, filename)
#             break
#         except:
#             errors.append(dict_item)
#             continue
#
#
# pickle.dump(errors, open('data/logs/errors_download_test.pkl', 'wb'))

import os

list_filename = os.listdir(folder_path)/8
list_filename = [filename.split('.')[0] for filename in list_filename]
list_imageid = [dict_item['imageId'] for dict_item in data]
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,}

diff = list(set(list_imageid)-set(list_filename))
assert len(list_filename)+len(diff)==len(list_imageid)

index = {k:v for v,k in enumerate(list_imageid)}

for diff_item in tqdm.tqdm(diff):
    dict_item = data[index[diff_item]]
    filename = '{}/{}.jpg'.format(folder_path, dict_item['imageId'])

    for url in dict_item['url']:
        #print(url, dict_item['imageId'])
        try:
            request=urllib.request.Request(url,None,headers)
            response = urllib.request.urlopen(request, timeout=500)
            f = open(filename, 'wb')
            f.write(response.read())
            f.close()
            break
        except:
            pass
