import os,numpy as np
from os.path import dirname, exists, join, splitext
import json,scipy
class Dataset(object):
    def __init__(self, dataset_name):
        self.work_dir = dirname(os.path.realpath('__file__'))
        info_path = join(self.work_dir, 'datasets', dataset_name + '.json')
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.palette = np.array(info['palette'], dtype=np.uint8)


def get_semantic_map(path):
    dataset=Dataset('cityscapes')
    semantic=scipy.misc.imread(path)
    tmp=np.zeros((semantic.shape[0],semantic.shape[1],dataset.palette.shape[0]),dtype=np.float32)
    for k in range(dataset.palette.shape[0]):
        tmp[:,:,k]=np.float32((semantic[:,:,0]==dataset.palette[k,0])&(semantic[:,:,1]==dataset.palette[k,1])&(semantic[:,:,2]==dataset.palette[k,2]))
    return tmp.reshape((1,)+tmp.shape)

def print_semantic_map(semantic,path):
    dataset=Dataset('cityscapes')
    semantic=semantic.transpose([1,2,3,0])
    prediction=np.argmax(semantic,axis=2)
    color_image=dataset.palette[prediction.ravel()].reshape((prediction.shape[0],prediction.shape[1],3))
    row,col,dump=np.where(np.sum(semantic,axis=2)==0)
    color_image[row,col,:]=0
    scipy.misc.imsave(path,color_image)
