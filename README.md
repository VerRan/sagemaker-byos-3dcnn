# 如何将开源项目迁移到SageMaker

在开始介绍将代码迁移到Sagemaker之前，首先简要介绍一下为什么要迁移代码到Sagemaker，大家都知道机器学习项目是需要大量的计算资源的，这里主要是因为机器学习一般都需要大量的数据通过复杂的计算来拟合一个表达来实现对新数据的预测能力，数据量，模型的复杂度都会影响模型的训练时间和模型效果，从模型训练时间角度来看，如果能够有大量的弹性扩展的资源来帮助模型训练，这样可以大大的缩减运维和资金成本，Sagemaker是Amazon提供的一个端到端的机器学习平台包括模数据预处理，模型训练，模型调试，模型理解，模型部署，模型监控，机器学习工作流自动化等能力的平台，本示例会重点介绍如何利用Sagemaker实现使用云上资源进数据预处理，模型训练。
 
## 背景
每当我们计划使用机器学习解决一个业务问题时，一般会采取如下的方案：
1. 当前业务问题，是否通过Saas的机器学习服务可以解决，比如图片的分类，可以考虑使用云服务提供商的SAAS服务，如AWS的Rekognition，通过API调用的方式可以快速实现自己的业务需求
2. 当方案1无法满足需求时，比如需要更多的定制化方案时，可以考虑使用比如Sagemaker的内置算法来降低算法开发和调优的复杂度。
3. 当方案2也无法满足需求时，比如需要自己来构建网络来解决业务问题，这个时候我们首先可以考虑在github上查找是否有已经经过验证的代码可以复用来降低复杂度和开发成本。同时为了更快速的搭建自己的机器学习环境以及实现更高效，更节约成本的方式来构建模型，此时可以考虑使用Sagemaker来搭建自己的机器学习环境实现这部分需求。
基于此背景，该文将介绍如何将搜索到的开源项目迁移到SageMaker，实现更高效，更节约成本的具体实现。

本文将从已下步骤进行介绍：
1. 业务理解并搜索需要的开源项目
2. 本地运行代码（使用Sagemaker notebook）
3. 迁移到Sagemaker的方案
3. 迁移模型到Sagemaker
4. 迁移预处理到Sagemaker

## 业务理解并搜索需要的开源项目
本文已通过3DCNN解决视频分类的问题来举例，比如当前您当前有一批视频需要基于视频的内容进行分类，需要将视频分为喜剧，动物，通过github搜索到一个3dcnn的网络（本示例采用 https://github.com/kcct-fujimotolab/3DCNN 代码），下来我们介绍如何具体进行迁移。

## 本地运行代码（使用Sagemaker notebook）
   在迁移之前首先我们将github上的代码下载到Sagemaker 笔记本实例中[如何开始使用sagemaker 笔记本实例](https://aws.amazon.com/cn/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)，然后通过sagemaker notebook对代码进行本地测试，当然也可以将代码下载到本机进行测试，通过Sagemaker notebook的优势在于可以快速的构建自己的开发环境，同时支持多种机型选择包括GPU机型，在后续的数据并行预处理以及模型的并行训练我们都会使用Sagemaker notebook来进行开发。
   下面对该项目的代码从数据预处理，模型训练两个部分进行简要介绍，这样便于后面我们的迁移改造。

### 数据预处理代码
如下代码用于将视频数据转换成np，对应3DCNN代码库中的pre_process.py类，目的是将视频数据转换为np后用于输入到网络中用于模型训练。

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

通过在notebook中执行process方法，代码会将视频数据（存储在dataset目录下的测试数据）通过opencv进行抽帧然后将其转换为numpy数组并最终将数据存储为.npz文件用于后续模型训练，预处理后的数据存储在default-output目录下。

当前我们已经准备好数据，为了确保下载的开源项目可以正常运行，先通过在notebook中进行本地测试的方式来验证。

### 本地训练模型
sagemaker-3dcnn.py 对应3DCNN代码库中的3dcnn.py，为了后续迁移到Sagemaker部分参数做了修改，后续会详细介绍。
```python
! python3 sagemaker-3dcnn.py --batch 3 --data_dir np-datasets --epoch 3 --output default-output  --nclass 8
```
通过执行如上代码，可以测试原有开源项目的网络是否可以正常运行并达到预期，在实际项目中会根据我们的需求进行优化，包括数据预处理，网络结构，超参数等，通过数据预处理阶段比如抽帧的间隔来提高样本的数量来提高模型效果，通过调整网络结构，超参数等来优化模型效果，此部分的优化会根据不同的数据情况和实际训练效果会采用不通的方式，此部分仅用于测试暂且不详细说明。

## 迁移到Sagemaker的方案
下来我们将介绍如何将如上3DCNN代码迁移到Sagemaker，针对迁移到Sagemaker有如下两种方案：
 1. BYOS（Bring Your Own Script），也就是说直接使用现有的网络代码并迁移到sagemaker。
 2. BYOC（Bring Your Own Container），也就是说将现有代码构建程自定义Docker镜像的方式迁移到Sagemaker
 
一般情况下建议使用BYOS的方案，当BYOS方案无法满足需求时比如需要使用您当前的环境代码和依赖直接迁移到Sagemaker该方案相较于BYOS需要自己构建镜像。
 
## 迁移模型到Sagemaker（BYOS）
 
  下来将介绍如何使用BYOS进行迁移代码，使用BYOS的方法和BYOC的不同之处在于：BYOC是使用用户自己创建的镜像来运行程序，更适用于用户对镜像自定义程度较高的使用情景；而BYOS是使用预先构建好的镜像，只是传入用户自己的代码来运行程序，不需要用户自己调试镜像，更适用于比较简单的使用情景。
  由于不需要编译自定义镜像，我们可以直接进行本地测试和Amazon SageMaker测试，完整流程见 sagemaker-3dcnn.ipynb。
  具体包含如下步骤：
  1. 引入依赖：需要引入Sagemaekr相关依赖
  2. 数据准备：Sagemaker模型训练会建议将数据存储到S3对象存储中，这里通过Saegmaker api可以快速将本地数据上传到S3。
  3. 超参数设置：在Sagemaker中代码的参数通过超参数设置来实现
  4. 模型代码修改：Sagemaker需要传递包括输入数据存储位置，模型的存储位置等，这里只需要修改模型的入口参数即可，针对模型本身的网络部分代码是没有侵入的。
  5. 使用Sagemaker进行模型训练：Sagemaker 提供了Estimator评估器来实现不同机器学习框架的封装从而简化对应框架的集成，比如通过TensorFlow Estimator实现了Tensorflow的封装，比如针对上文使用的3dcnn项目只需要将修改参数的3dcnn.py作为参数传递给TensorFlowEstimator就可以自动实现使用Sagmaker来实现模型的训练，同时使用到Sagemaker强大的资源调度和管理的能力包括多种机型的支持，以及CPU，GPU训练，并行训练等。通过并行训练可以有效的提升训练速度，下文数据预处理章节就采用了并行方式。
  
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

### 数据准备
当使用Sagemaker进行数据预处理或者模型训练时，建议将数据上传到S3，这样便于针对大数据量场景下的数据预处理和模型训练，下面我们将预处理后的数据上传到S3。


```python
# 将np-datasets目录下的np文件上传到sagemaker-studio默认桶下的np-datasets文件夹下
inputs = sagemaker.Session().upload_data(path='np-datasets', key_prefix='np-datasets')
inputs
```

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

### 模型代码修改
由于Sagemaker会自动通过S3下载训练数据到模型训练的机器，然后使用下载的数据进行模型训练，同时也会自动将训练好的模型上传到S3中，因此我们需要对原有的代码进行调整，添加如下参数：
1. model_dir：用于指定模型的S3存储路径，如果不设置的话默认会存储到Sagemaker studio默认创建的桶中
2. data_dir： 用于指定模型训练的训练数据存储路径，路径对应fit方法的训练输入路径参数

* parser.add_argument('--model_dir', type=str)
* parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))

### 使用Sagemaker 实现模型的训练
Sgamaker实现模型训练有两种API使用方式：
1. Sagemaker low level的API 也就是Sgaemaker 的API，这个是针对Sagemaker服务开放的API支持多种语言如Python java具体参考[Sagamker API](https://docs.amazonaws.cn/sagemaker/latest/dg/api-and-sdk-reference.html)
2. Sagemaker high level的API 也就是[Sagemaker的Python SDK](https://sagemaker.readthedocs.io/en/stable/)
使用Sagemaker high level的API更加简单和强大，一般会建议使用该方案。下来我们使用Sagemaker Python SDK提供的TensoFlow 评估器(Estimator)来实现。


```python
estimator = TensorFlow(entry_point='sagemaker-3dcnn.py',
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

当模型训练成功后，模型会存储到S3中，可以通过Sagemaker控制台找到此次训练的任务，在详情页面可以找到模型存储路径，样本数据存储路径，以及我们上面设置的超参数信息。

## 迁移预处理到Sagemaker并优化（BYOC）

上面我们在数据预处理章节采用notebook 实例对测试数据进行了预处理，如果样本数据比较少的话notebook实例可以用于测试，但是当样本数据的量级很大的时候无论是从存储能力还是处理能力notebook都是无法满足的呢。针对预处理阶段Sagemaker提供了Sagemaker Processing功能用于解决如上问题，下文将通过Sagemaker Processing 来介绍如何实现：
1. 如何实现存储在S3中的大批量数据进行预处理
2. 当处理性能不能满足时，如何提高处理性能

### Sagemaker Processing 处理存储在S3中的数据 

Sagemaker Processing 实际上也是在后台运行了一个job用于调度计算资源来运行我们定义好的预处理脚本来进行数据预处理。本文将已上面视频数据预处理为例子将上面的预处理代码迁移到Sagemaker Processing中。

在Sagemaker processing 的迁移中也有BYOS和BYOC两种方案，下面我们使用BYOC的方式来迁移预处理代码。

### 迁移代码到Sageamker Processing（BYOC）
1. 需要通过ECR创建一个镜像仓库，用于后续存储自定义的数据预处理镜像
2. 构建自定义镜像，并上传存储库
3. 通过notebook运行Sagemaker processing job来运行任务

#### 使用ECR创建仓库
1. 使用ECR仓库提示的命令进行镜像构建和上传（https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-console.html）
2. 进入container文件夹下，运行步骤1中ECR提示命令构建镜像，并上传到ECR
Dockerfile中定义了数据预处理代码以及相关依赖代码，并定义了运行入口代码，下面是具体的docker定义文件：
* RUN ： 运行pip命令安装所需依赖
* RUN mkdir ：创建dataset文件夹
* ADD ：添加自定义的依赖到容器中
* ENTRYPOINT：指定容器运行的入口代码


```python
FROM python:3.7-slim-buster

RUN pip3 install keras tensorflow opencv-python-headless tqdm numpy
RUN mkdir dataset
ADD processing_script.py  videoto3d.py classes.txt / 
ENTRYPOINT ["python3", "/processing_script.py"]
```

通过如下命令构建自定镜像，并上传镜像到ECR中


```python
docker build -t test .
docker tag test:latest xxx.dkr.ecr.cn-northwest-1.amazonaws.com.cn/test:latest
docker push xxx.dkr.ecr.cn-northwest-1.amazonaws.com.cn/test:latest
```

定义ECR中自定义的镜像地址，用于Sagemaker Processing job进行饮用

#### 使用自定义镜像运行Sagemaker Processing job

自定义镜像地址定义，用于ProcessingJob引用并进行模型训练


```python
image_uri = 'xx.dkr.ecr.cn-northwest-1.amazonaws.com.cn/test:latest'
```

定义Process，并启动任务。


```python
processor = Processor(image_uri=image_uri,
                     role=role,
                     instance_count=1,
                     instance_type="ml.c5.2xlarge")
processor.run(inputs=[ProcessingInput(
                        source='s3://sagemaker-us-west-2-517141035927/dataset/videos/',
                        destination='/opt/ml/processing/input_data')],
                    outputs=[ProcessingOutput(
                        source='/opt/ml/processing/processed_data',
                        destination='s3://sagemaker-us-west-2-517141035927/dataset/np/')],
                    )
```

## 总结
本文通过一个开源项目的示例介绍了以下内容：
1. 如何将自定义的网络或者第三方的网络迁移到Sagemaker（BYOS方式）
2. 如何将预处理代码迁移到Sagemaker进行数据预处理（BYOC方式）

源代码地址：https://github.com/VerRan/sagemaker-byos-3dcnn
