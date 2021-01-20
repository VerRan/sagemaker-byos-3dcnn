# 使用自定义镜像进行数据预处理
* ECR 创建仓库
* 使用ECR仓库提示的命令进行镜像构建和上传[ECR入门](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-console.html)
* 打开Video_process.ipynb 根据步骤执行，这里使用Sagemaker Processing 来进行数据处理

* 拷贝npz文件到本地：aws s3 sync s3://sagemaker-us-west-2-517141035927/dataset/np/ np

## 参考
https://docs.aws.amazon.com/sagemaker/latest/dg/build-your-own-processing-container.html
