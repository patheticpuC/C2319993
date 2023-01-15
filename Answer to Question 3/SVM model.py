import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from torchvision import models, transforms
from sklearn.svm import LinearSVC

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(448),
    transforms.ToTensor()
])

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        self.pooling = model.avgpool
        self.flatten = nn.Flatten()
        self.fc = model.classifier[0]

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

def readImg(dir):
    img_list = []
    label_list = []
    for i in os.listdir(dir):
        for img in os.listdir(os.path.join(dir, i)):
            img_list.append(os.path.join(dir, i, img))
            label_list.append(i)
    return img_list, label_list

def features_img(lis_path):
    features_matrix = []
    count = 0
    for i in lis_path:
        img = cv2.imread(i)
        img = transform(img)
        img = img.reshape(1, 3, 448, 448)
        img = img.to(device)
        with torch.no_grad():
            feature = new_model(img)
        features_matrix.append(feature.cpu().detach().numpy().reshape(-1))
        count = count+1
        print(count)
    features_matrix = np.array(features_matrix)
    return features_matrix

model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

train_dir = r'C:\Users\ww\PycharmProjects\pythonProject3\dataset\train'
test_dir = r'C:\Users\ww\PycharmProjects\pythonProject3\dataset\test'
train_img, train_label = readImg(train_dir)
test_img, test_label = readImg(test_dir)
train_extract = features_img(train_img)
test_extract = features_img(test_img)

import time
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

lsvm = LinearSVC(C=5.0)
lsvm.fit(train_extract, train_label)
print("Linear_score")
print(lsvm.score(test_extract, test_label))
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

rsvm = SVC(kernel='rbf', C=5.0, gamma='auto')
rsvm.fit(train_extract, train_label)
print("rbf_score")
print(rsvm.score(test_extract, test_label))
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

rsvm2 = SVC(kernel='poly', C=5.0, gamma='auto')
rsvm2.fit(train_extract, train_label)
print("poly_score")
print(rsvm2.score(test_extract, test_label))
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

pca = PCA(0.99)
pca.fit(train_extract)
train_extract_reduction = pca.transform(train_extract)
test_extract_reduction = pca.transform(test_extract)
print(train_extract.shape)
print(train_extract_reduction.shape)

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

rsvm3 = SVC(kernel='rbf', C=5.0, gamma='auto')
rsvm3.fit(train_extract_reduction, train_label)
print("rbf3_score")
print(rsvm3.score(test_extract_reduction, test_label))

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

lsvm2 = LinearSVC(C=5.0)
lsvm2.fit(train_extract_reduction, train_label)
print("Linear_score")
print(lsvm2.score(test_extract_reduction, test_label))
