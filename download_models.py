import urllib.request

print('Dowloading VGG-19 Model (510Mb)')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/VGG_Model/imagenet-vgg-verydeep-19.mat','VGG_Model/imagenet-vgg-verydeep-19.mat')

print('Dowloading CRN 1024p Model (500Mb)')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_1024p/model.ckpt.data-00000-of-00001','result_1024p/model.ckpt.data-00000-of-00001')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_1024p/model.ckpt.meta','result_1024p/model.ckpt.meta')

print('Dowloading CRN 512p Model (1.2Gb)')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_512p/model.ckpt.data-00000-of-00001','result_512p/model.ckpt.data-00000-of-00001')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_512p/model.ckpt.meta','result_512p/model.ckpt.meta')

print('Dowloading CRN 256p Model (1.2Gb)')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_256p/model.ckpt.data-00000-of-00001','result_256p/model.ckpt.data-00000-of-00001')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_256p/model.ckpt.meta','result_256p/model.ckpt.meta')

print('Downloading GTA 256p Model (1.2Gb)')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_GTA/model.ckpt.data-00000-of-00001','result_GTA/model.ckpt.data-00000-of-00001')
urllib.request.urlretrieve('https://cqf.io/data/Tensorflow_models/result_GTA/model.ckpt.meta','result_GTA/model.ckpt.meta')
