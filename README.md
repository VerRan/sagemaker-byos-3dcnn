# 如何将开源项目迁移到SageMaker

## 背景
每当我们计划使用机器学习解决一个业务问题时，一般会采取如下的方案：
1. 当前业务问题，是否通过Saas的机器学习服务可以解决，比如图片的分类，可以考虑使用云服务提供商的SAAS服务，如AWS的Rekognition，通过API调用的方式可以快速实现自己的业务需求
2. 当方案1无法满足需求时，比如需要更多的定制化方案时，可以考虑使用比如Sagemaker的内置算法来降低算法开发和调优的复杂度。
3. 当方案2也无法满足需求时，比如需要自己来构建网络来解决业务问题，这个时候我们首先可以考虑在github上查找是否有已经经过验证的代码可以复用来降低复杂度和开发成本。同时为了更快速的搭建自己的机器学习环境以及实现更高效，更节约成本的方式来构建模型，此时可以考虑使用Sagemaker来搭建自己的机器学习环境实现这部分需求。
基于此背景，该文将介绍如何将搜索到的开源项目迁移到SageMaker，实现更高效，更节约成本的具体实现。

本文将从已下步骤进行介绍：
1. 业务理解并搜索需要的开源项目
2. Sagemaker notebook中运行代码
3. 编写迁移到Sagemaker的notebook代码
4. 预处理优化
5. 训练优化

## 业务理解并搜索需要的开源项目
本文已通过3DCNN解决视频分类的问题来举例，比如当前您当前有一批视频需要基于视频的内容进行分类，需要将视频分为喜剧，动物，通过github搜索到一个3dcnn的网络，下来我们介绍如何具体进行迁移。

## Sagemaker notebook中运行代码
1. 在迁移之前首先我们将github上的代码下载到Sagemaker studio中[如何开始使用sagemaker studio](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/studio.html)
2. 通过Sagemaker studio 开始运行代码进行测试

### 引入依赖


```python
import sagemaker
from sagemaker.tensorflow import TensorFlow
import os 
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()
region = sagemaker_session.boto_session.region_name

```

### 数据预处理
将视频数据转换成np，下面的代码介绍如何将视频数据转换为np用于输入到网络中用于模型训练


```python
from os import *
import codecs
from keras.utils import np_utils
import numpy as np
import videoto3d 


def loaddata(video_dir, vid3d, nclass, result_dir, color=False, skip=True):
        files = os.listdir(video_dir)
        X = []
        labels = []
        labellist = []

        for filename in files:
            if filename == '.DS_Store':
                continue
            if os.path.splitext(filename)[1] != ".mp4":
                continue
            name = os.path.join(video_dir, filename)
#             print('filename is11 ',filename)
            label = vid3d.get_UCF_classname(filename)
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(vid3d.video3d(name, color=color, skip=skip))
            
        print('result_dir is ',result_dir)
        fpath = result_dir + 'classes.txt'
        fp = codecs.open(fpath,'a','utf-8')
        print('labels length is ',len(labellist))
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

        for num, label in enumerate(labellist):
            for i in range(len(labels)):
                if label == labels[i]:
                    labels[i] = num
        if color:
            return np.array(X).transpose((0, 2, 3, 4, 1)), labels
        else:
            return np.array(X).transpose((0, 2, 3, 1)), labels

        
def process():
        nclass = 8
        depth = 15
        skip = False
        color = True
        img_rows, img_cols, frames = 32, 32, depth

        channel = 3 if color else 1
        fname_npz = 'np-datasets/train_data.npz'
        output = 'default-output/'
        videos = 'dataset/'

        vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
        nb_classes = nclass
        if os.path.exists(fname_npz):
                loadeddata = np.load(fname_npz)
                X, Y = loadeddata["X"], loadeddata["Y"]
        else:
                x, y = loaddata(videos, vid3d, nclass,
                                output, color, skip)
                X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
                Y = np_utils.to_categorical(y, nb_classes)

                X = X.astype('float32')
                np.savez(fname_npz, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')
        print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

```


```python
process()
```

    result_dir is  default-output/
    labels length is  4
    Saved dataset to dataset.npz.
    X_shape:(4, 32, 32, 15, 3)
    Y_shape:(4, 8)


如上代码会将视频数据（存储在dataset目录下的测试数据）通过opencv进行抽帧然后将其转换为numpy数组并最终将数据存储为.npz文件用于后续模型训练，预处理后的数据存储在default-output目录下

下来我们在notebook中测试训练网络


```python
# ! python3 sagemaker-3dcnn.py --batch 3 --data_dir np-datasets --epoch 3 --output default-output  --nclass 8
```

当前使用测试数据可以跑通网络，由于数据仅仅用于测试这里针对模型的效果暂时不做处理。

 ## 编写迁移到Sagemaker的notebook代码
 下来我们将介绍如何将如上运行通过的代码迁移到Sagemaker，针对迁移到Sagemaker有如下两种方案：
 1. BYOS（Bring Your Own Script），也就是说直接使用现有的网络代码并迁移到sagemaker
 2. BYOC（Bring Your Own Container），也就是说将现有代码构建程自定义Docker镜像的方式迁移到Sagemaker
 一般情况下建议使用BYOS的方案，该方案更加简单，当BYOS方案无法满足需求时比如需要使用您当前的环境代码和依赖直接迁移到Sagemaker该方案相较于BYOS需要自己构建镜像。
 本文将介绍如何使用BYOS进行迁移。

### 数据准备
当使用Sagemaker进行数据预处理或者模型训练时，建议将数据上传到S3，这样便于针对大数据量场景下的数据预处理和模型训练，下面我们将预处理后的数据上传到S3。


```python
# 将np-datasets目录下的np文件上传到sagemaker-studio默认桶下的np-datasets文件夹下
inputs = sagemaker.Session().upload_data(path='np-datasets', key_prefix='np-datasets')
inputs
```




    's3://sagemaker-cn-northwest-1-462130072016/np-datasets'



打开S3控制台查看如上输出路径可以看到预处理后的数据已经上传到S3中了。

### 超参数设置
在本地运行的时候网络的参数是通过 python命令行传入的，当迁移到Sagemaker后可以通过参数的方式设置超参数，然后传递给Sagemaker进行使用。
参数说明：
1. data_dir：指定训练数据从S3下载后存储到训练机器上的路径
2. output：指定模型训练完成后模型存储在训练机器上的路径
3. epoch/batch/nclass：此部分参数是原有网络中使用的参数，提取出来通过超参数传递给网络


```python
hyperparameters = {'epoch': 3, 
                   'data_dir': '/opt/ml/input/data/training',
                   'batch': 3, 
                   'nclass': 8,
                   'output': '/opt/ml/output',
                  }
```

### 代码修改
由于Sagemaker会自动通过S3下载训练数据到模型训练的机器，然后使用下载的数据进行模型训练，同时也会自动将训练好的模型上传到S3中，因此我们需要对原有的代码进行调整，添加如下参数：
1. model_dir：用于指定模型的S3存储路径，如果不设置的话默认会存储到Sagemaker studio默认创建的桶中
2. data_dir： 用于指定模型训练的训练数据存储路径，路径对应fit方法的训练输入路径参数

* parser.add_argument('--model_dir', type=str)
* parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))

### 使用Sagemaker Python SDK API 实现模型的训练
Sgamaker的有两种方式：
1. Sagemaker low level的API 也就是Sgaemaker 的API，这个是针对Sagemaker服务开放的API支持多种语言如Python java具体参考[Sagamker API](https://docs.amazonaws.cn/sagemaker/latest/dg/api-and-sdk-reference.html)
2. Sagemaker high level的API 也就是[Sagemaker的Python SDK](https://sagemaker.readthedocs.io/en/stable/)
使用Sagemaker high level的API更加简单和强大，一般会建议使用该方案。下来我们使用Sagemaker Python SDK提供的TensoFlow 评估器(Estimator)来实现。


```python
# image_uri = '727897471807.dkr.ecr.cn-northwest-1.amazonaws.com/tensorflow-training:1.15.4-gpu-py37-cu100-ubuntu18.04'
# ## Deep Learning Container 说明地址：https://github.com/aws/deep-learning-containers
# ## DLC 地址 ：https://github.com/aws/deep-learning-containers/blob/master/available_images.md 可用列表

```


```python
estimator = TensorFlow(entry_point='sagemaker-3dcnn.py',
#                        image_uri = image_uri,
                       train_instance_type='ml.c4.xlarge',
                       train_instance_count=1,
                       hyperparameters=hyperparameters,
                       role=sagemaker.get_execution_role(),
                       framework_version='1.15.2',
                       py_version='py3',
                       script_mode=True)
```


```python
estimator.fit({'training': inputs})
```

    'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.
    's3_input' class will be renamed to 'TrainingInput' in SageMaker Python SDK v2.
    'create_image_uri' will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.


    2021-01-20 10:25:10 Starting - Starting the training job..


```python

```
