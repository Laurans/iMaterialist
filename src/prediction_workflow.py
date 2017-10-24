import pandas as pd
import json
import os
import pickle
from torch.autograd import Variable
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# df = pd.read_csv('./data/raw/fgvc4_iMat.test.samplesubmission.csv')
# df['imageId'], df['taskId'] = zip(*df.id.str.split('_'))

# with open('data/raw/fgvc4_iMat.task_map.json') as data_file:
#     task_map = json.load(data_file)['taskInfo']
#     task_map = {d['taskId']: d['taskName'] for d in task_map}
#
# list_filename = os.listdir('./data/test_raw')
# list_filename = [filename.split('.')[0] for filename in list_filename]
#
# df = df[df.imageId.isin(list_filename)]
# df['taskName'] = df.taskId.apply(lambda x: task_map[x])
# df['apparel'], df['attribute'] = zip(*df.taskName.str.split(':'))
# df['path'] = df.imageId.apply(lambda x: '../data/{}/{}.jpg'.format('test_raw', x))

def prediction(trained_model):
    dset = pickle.load( open('../data/serialized/test.pkl', 'rb'))
    bs = 64
    dset_loaders = DataLoader(dset, batch_size=bs, shuffle=False, num_workers=4)
    items = []
    print('Start prediction')
    for i, data in enumerate(tqdm(dset_loaders, ncols=50, ascii=True)):
            inputs_tensors, labels_tensors = data
            # wrap them in Variable
            inputs = list(map(lambda x: Variable(x.cuda()), inputs_tensors))
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs[0].data,1)
            preds = preds.cpu().numpy().flatten()
            l = [('{}_{}'.format(path.replace('../data/test_raw/', '').replace('.jpg', ''), task), preds[k]) for k, (path, _, _, task, _) in enumerate(dset.imgs[bs*i:bs*(i+1)])]
            items.extend(l)
            #for k, (path, task, _, _) in enumerate(dset.imgs[bs*i:bs*(i+1)]):
            #    img = path.replace('../data/test_raw/', '').replace('.jpg', '')
            #    my_item = ('{}_{}'.format(img, task), preds[k])
            #    items.append(my_item)

    pickle.dump(items, open('../result_for_submission.pkl', 'wb'))
    print('Making csv submission...')
    sub = pd.read_csv('../data/raw/fgvc4_iMat.test.samplesubmission.csv')
    df = pd.DataFrame(items, columns=sub.columns)
    s1 = sub.id.values
    s2 = df.id.values
    diff = set(s1) - set(s2)
    part = sub[sub.id.isin(diff)]
    df2 = pd.concat([df, part])
    df2['idd'], df2['task'] = zip(*df2.id.str.split('_'))
    df2.idd = pd.to_numeric(df2.idd)
    df2.task = pd.to_numeric(df2.task)
    df3 = df2.sort_values(by=['idd', 'task'], ascending=[True, True])
    df3[['id', 'predicted']].to_csv('./my_submission.csv', index=False)

