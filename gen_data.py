# 自己写的用于提取数据的代码
# 基本思想就是以180个timestep为一小块对原视频进行划分，每一小块输入模型并得到最后的结果
# 即每一小块就是一个数据样本

import os
import sys
import pickle
import numpy as np
import argparse
from numpy.lib.format import open_memmap
import scipy.io as sio
from collections import Counter
import random

num_joint = 22  # 总共22个关节
max_frame = 180  # 是按3s来对数据进行切割的，而摄像机频率为60Hz，故一个窗口内有180个timestep


def gauss_noise(data, var=0.05):
    # data:N*180*66
    '''
    这里是以数据均值作为高斯的均值的，但是不是这样的，高斯的均值应该为0
    dataWithNoise = np.zeros(data.shape)
    for i in range(data.shape[2]):
        temp_coor = data[:,:,i]
        mean = np.mean(temp_coor)
        noise = np.random.normal(mean,var,mean.shape)
        dataWithNoise[:,:,i] = noise + temp_coor
    return dataWithNoise
    '''
    dataWithNoise = np.zeros(data.shape)
    for i in range(data.shape[2]):
        temp_coor = data[:, :, i]
        noise = np.random.normal(0, var, temp_coor.shape)
        dataWithNoise[:, :, i] = noise + temp_coor
    return dataWithNoise


def cropping(data, prob=0.05):
    # 总共需要丢弃totaltimestep*prob的数据
    i = 0
    window_num = data.shape[0]
    window_size = data.shape[1]
    random1 = 0
    random2 = 0
    cropped_data = data
    while i < window_num * window_size * prob:
        random1 = random.randint(0, window_num-1)
        random2 = random.randint(0, window_size-1)
        if ~(np.any(cropped_data[random1, random2, :]) == 0):
            cropped_data[random1, random2, :] = 0
            i = i + 1
    return cropped_data
    
def gendata(data_path,
            out_path,
            mode='train'):
    x_total = np.zeros((1, 66))
    y_total = np.zeros((1, 3))
    for filename in os.listdir(data_path):
        dataframe = sio.loadmat('{}/{}'.format(data_path, filename))
        data = dataframe['data']

        # len用于记录timestep大小
        len = data.shape[0]
        i = 0
        # 180是因为一个切片有180个timestep，0.5是重叠概率为0.5
        # 区间左端点为i*90，右端点为180+i*90
        while (180 + i * 90 < len):
            x = data[i * 90:180 + i * 90, :66]
            # 取出这一切片中出现频率最高的作为疼痛强度
            y = Counter(data[i * 90:180 + i * 90, 70]).most_common(1)[0][0]
            # 这里需要加一以将-1调整到0
            y2 = Counter(data[i * 90:180 + i * 90, 71]).most_common(1)[0][0] + 1
            y3 = Counter(data[i * 90:180 + i * 90, 72]).most_common(1)[0][0]
            y_sum = np.append(y, y2)
            y_sum = np.append(y_sum, y3)

            y_sum = np.expand_dims(y_sum, 0)
            x_total = np.concatenate((x_total, x), 0)
            y_total = np.concatenate((y_total, y_sum), 0)

            i = i + 1
    # 所有数据都在x_total和y_total中
    x_total = x_total[1:, :]
    # 需要将这里reshape成-1，180，66

    x_total = x_total.reshape(-1, 180, 66)
    y_total = y_total.reshape(-1, 3)
    y_total = y_total[1:, :]

    # 数据扩充
    '''
        p1即原数据
        p2为施加了0.05标准差的高斯噪声
        p3为施加了0.1标准差的高斯噪声
        p4为百分之5的几率置0的cropping
        p5为百分之10的几率置0的cropping
    '''
    if mode == 'train':
        # 对训练样本添加正态和椒盐噪声以扩充数据
        x_p2 = gauss_noise(x_total, var=0.05)
        x_p3 = gauss_noise(x_total, var=0.1)
        x_p4 = cropping(x_total, prob=0.05)
        x_p5 = cropping(x_total, prob=0.1)
        print(x_total.shape)
        y_p2 = y_total
        y_p3 = y_total
        y_p4 = y_total
        y_p5 = y_total

        x_total = np.concatenate((x_total, x_p2, x_p3, x_p4, x_p5), 0)
        y_total = np.concatenate((y_total, y_p2, y_p3, y_p4, y_p5), 0)
    
    print(x_total.shape)
    print(y_total.shape)
    
    fp = open_memmap('{}_data.npy'.format(out_path),
                     dtype='float32',
                     mode='w+',
                     shape=(x_total.shape[0], 180, 66))
    fp[:, :, :] = x_total
    fp2 = open_memmap('{}_label.npy'.format(out_path),
                      dtype='float32',
                      mode='w+',
                      shape=(y_total.shape[0], 3))
    fp2[:] = y_total
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='emo-pain Data Converter.')
    parser.add_argument('--data_path', default='./raw_data')
    parser.add_argument('--out_path', default='./cooked_data')

    arg = parser.parse_args()
    if not os.path.exists(arg.out_path):
        os.makedirs(arg.out_path)
    part = ['train', 'eval']
    for p in part:
        gendata('{}/{}'.format(arg.data_path, p),
                '{}/{}'.format(arg.out_path, p),
                mode=p)
