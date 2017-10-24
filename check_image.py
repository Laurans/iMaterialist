from PIL import Image
import json
import urllib.request
import tqdm
import os
import shutil

import warnings
warnings.filterwarnings('error')

folder_path = 'data/train_raw'
with open('data/raw/fgvc4_iMat.train.data.json') as data_file:
    data = json.load(data_file)
    data = data['images']

imageId_map  = {'{}.jpg'.format(dict_item['imageId']):dict_item['url'] for dict_item in data}
list_filename = os.listdir(folder_path)


errors = []
for filename in list_filename:
    try:
        with Image.open("{}/{}".format(folder_path, filename)) as im:
            i = im.convert('RGB')
    except:
        errors.append(filename)


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,}

for error in tqdm.tqdm(errors):
    if len(imageId_map[error]) > 2:
        for url in imageId_map[error][2:]:

            try:
                request=urllib.request.Request(url,None,headers)
                response = urllib.request.urlopen(request, timeout=500)
                f = open("{}/{}".format(folder_path,error), 'wb')
                f.write(response.read())
                f.close()
                break
            except:
                pass
    else:
        os.remove("{}/{}".format(folder_path,error))
