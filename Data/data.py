import scipy.io
import torch
import numpy as np


def load_mat_dataset(mat_file_path = r'C:\Users\Kusha\OneDrive\Desktop\LLM\DRAW\Data\dataset\train_32x32.mat'):
    mat_data = scipy.io.loadmat(mat_file_path)

    X = mat_data['X']  
    labels = mat_data['y']

    labels = labels.squeeze()
    labels[labels == 10] = 0

    return X,labels
    
def image_with_label_list(x, y):
    data_list = []
    for i in range(len(y)):
        image = x[:, :, :, i]
        label = y[i]
        data_list.append((image, label))
    return data_list

if __name__ == "__main__":
    X, labels = load_mat_dataset()
    data_list = image_with_label_list(X, labels)