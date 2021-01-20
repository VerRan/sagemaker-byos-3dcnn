import os
import numpy as np
# 拷贝npz文件到本地：aws s3 sync s3://sagemaker-us-west-2-517141035927/dataset/np/ np 用于测试
base_dir = './np'
files = os.listdir(base_dir)
nps = []
for f in files:
	print(f)
	np_data = np.load(os.path.join(base_dir, f))
	nps.append(np_data)
np.savez('train.npz',nps)