
import glob

import numpy as np

len_list = []
for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Clean_Data/' + '*.png'):
    len_list.append(len(img))

Slen_list = np.unique(len_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Clean_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)
    if (len(img) == Slen_list[3]):
        DVec4.append(img)

DVec_SortOrig = []
DVec_SortOrig.extend(sorted(DVec1))
DVec_SortOrig.extend(sorted(DVec2))
DVec_SortOrig.extend(sorted(DVec3))
DVec_SortOrig.extend(sorted(DVec4))

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Clean_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img[len(img)-25:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[1]):
        DVec2.append(img[len(img)-26:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[2]):
        DVec3.append(img[len(img)-27:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[3]):
        DVec4.append(img[len(img)-28:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])

DVec_SortNew = []
DVec_SortNew.extend(sorted(DVec1))
DVec_SortNew.extend(sorted(DVec2))
DVec_SortNew.extend(sorted(DVec3))
DVec_SortNew.extend(sorted(DVec4))

import os
os.chdir('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Clean_Data')

for ig in range(0,len(DVec_SortOrig)):
    orig_temp = DVec_SortOrig[ig]
    new_temp = DVec_SortNew[ig]
    os.rename(orig_temp,new_temp)

import glob

import numpy as np

len_list = []
for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Data/' + '*.png'):
    len_list.append(len(img))

Slen_list = np.unique(len_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)
    if (len(img) == Slen_list[3]):
        DVec4.append(img)

DVec_SortOrig = []
DVec_SortOrig.extend(sorted(DVec1))
DVec_SortOrig.extend(sorted(DVec2))
DVec_SortOrig.extend(sorted(DVec3))
DVec_SortOrig.extend(sorted(DVec4))

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img[len(img)-25:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[1]):
        DVec2.append(img[len(img)-26:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[2]):
        DVec3.append(img[len(img)-27:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[3]):
        DVec4.append(img[len(img)-28:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])

DVec_SortNew = []
DVec_SortNew.extend(sorted(DVec1))
DVec_SortNew.extend(sorted(DVec2))
DVec_SortNew.extend(sorted(DVec3))
DVec_SortNew.extend(sorted(DVec4))

import os
os.chdir('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Fooling_Data')

for ig in range(0,len(DVec_SortOrig)):
    orig_temp = DVec_SortOrig[ig]
    new_temp = DVec_SortNew[ig]
    os.rename(orig_temp,new_temp)


import glob

import numpy as np

len_list = []
for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Clean_Data/' + '*.png'):
    len_list.append(len(img))

Slen_list = np.unique(len_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Clean_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)
    if (len(img) == Slen_list[3]):
        DVec4.append(img)

DVec_SortOrig = []
DVec_SortOrig.extend(sorted(DVec1))
DVec_SortOrig.extend(sorted(DVec2))
DVec_SortOrig.extend(sorted(DVec3))
DVec_SortOrig.extend(sorted(DVec4))

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Clean_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img[len(img)-25:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[1]):
        DVec2.append(img[len(img)-26:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[2]):
        DVec3.append(img[len(img)-27:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[3]):
        DVec4.append(img[len(img)-28:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])

DVec_SortNew = []
DVec_SortNew.extend(sorted(DVec1))
DVec_SortNew.extend(sorted(DVec2))
DVec_SortNew.extend(sorted(DVec3))
DVec_SortNew.extend(sorted(DVec4))

import os
os.chdir('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Clean_Data')

for ig in range(0,len(DVec_SortOrig)):
    orig_temp = DVec_SortOrig[ig]
    new_temp = DVec_SortNew[ig]
    os.rename(orig_temp,new_temp)


import glob

import numpy as np

len_list = []
for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Data/' + '*.png'):
    len_list.append(len(img))

Slen_list = np.unique(len_list)

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img)
    if (len(img) == Slen_list[1]):
        DVec2.append(img)
    if (len(img) == Slen_list[2]):
        DVec3.append(img)
    if (len(img) == Slen_list[3]):
        DVec4.append(img)

DVec_SortOrig = []
DVec_SortOrig.extend(sorted(DVec1))
DVec_SortOrig.extend(sorted(DVec2))
DVec_SortOrig.extend(sorted(DVec3))
DVec_SortOrig.extend(sorted(DVec4))

DVec1 = []
DVec2 = []
DVec3 = []
DVec4 = []

for img in glob.glob('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Data/' + '*.png'):
    if (len(img) == Slen_list[0]):
        DVec1.append(img[len(img)-25:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[1]):
        DVec2.append(img[len(img)-26:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[2]):
        DVec3.append(img[len(img)-27:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])
    if (len(img) == Slen_list[3]):
        DVec4.append(img[len(img)-28:len(img)-17] + 'deepfool' + img[len(img)-13:len(img)])

DVec_SortNew = []
DVec_SortNew.extend(sorted(DVec1))
DVec_SortNew.extend(sorted(DVec2))
DVec_SortNew.extend(sorted(DVec3))
DVec_SortNew.extend(sorted(DVec4))

import os
os.chdir('/home/shayan/CIFAR/Adv_Train/Backup/DeepFool/Unfooling_Data')

for ig in range(0,len(DVec_SortOrig)):
    orig_temp = DVec_SortOrig[ig]
    new_temp = DVec_SortNew[ig]
    os.rename(orig_temp,new_temp)
