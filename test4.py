import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import glob

# 修改文件名

def rename(path):
    # path = "E:/Project/Saliency_Matlab/debug/result_mask/test_res/GMR/"
    fileList = os.listdir(path)
    counter = 0
    for item in fileList:
        oldname = path + fileList[counter]
        print('old===' + item)
        len_num = len(item)
        if (item.find('_BSCA') >=0):
            newname = path + item[:len_num-9] + '.png'
            print('new  == '+newname)
            os.rename(oldname,newname)
            print(newname)
        counter += 1


def prSingle(img_path, mask_path):
    fileList_img = os.listdir(img_path)
    fileList_mask = os.listdir(mask_path)
    for j in range(1):
        sum_fp_tp = 0
        sum_fn_tp = 0
        tp = 0
        for i in range(len(fileList_mask)):
            # print("num = :", len(fileList_mask))
            filePath = os.path.join(img_path, fileList_img[i])
            if os.path.isfile(filePath):
                print(filePath)
                pic = cv2.imread(filePath, 0)
                imgNp = np.array(pic)
                # imgNp = np.where((imgNp) >= j, 255, 0)
                sum_fp_tp = np.sum(imgNp / 255)
            filePath_mask = os.path.join(mask_path, fileList_mask[i])
            if os.path.isfile(filePath_mask):
                print(filePath_mask)
                pic_mask = cv2.imread(filePath_mask, 0)
                maskNp = np.array(pic_mask)
                sum_fn_tp = np.sum(maskNp / 255)
                tp_test = np.multiply(imgNp / 255, maskNp / 255)
                tp = np.sum(tp_test)
                # print("tp == ", tp, " sum_img==", sum_fp_tp, " sum_mask == ", sum_fn_tp)
        precise= tp / sum_fp_tp
        recall = tp / sum_fn_tp
        beta = 0.3
        beta = np.square(beta)
        F = (1 + beta) * precise * recall / (beta * precise + recall)
        print(img_path)
        print("num = ",j, "final precise =  ", precise,"final recall == ", recall)
    return precise, recall,F



def F(precise,recall):
    beta = 0.3
    beta = np.square(beta)
    F = list()
    for i in range(len(precise)):
        F.append ((1+ beta) * precise[i] * recall[i] / (beta * precise[i] + recall[i]))
    return F

def MAE(sourcePath, GTPath):
    imgPath = os.listdir(sourcePath)
    gtPath = os.listdir(GTPath)
    imNum = len(imgPath)
    mae = 0
    for i in range(imNum):
        img = cv2.imread(sourcePath + imgPath[i])
        # print(sourcePath + imgPath[i])
        gt = cv2.imread(GTPath + gtPath[i])
        # cv2.imshow("22",gt)
        # cv2.waitKey(0)
        imgNp = np.array(img)
        gtNp = np.array(gt)
        diff = (imgNp - gtNp)
        diff = np.where(diff < 0 ,-diff,diff)
        mae = mae + (1.0 / np.size(diff)) * (np.sum(diff/255))
    mae = (mae / imNum)
    print(mae)
    return mae

def MOR(sourcePath, GTPath):
    imgPath = os.listdir(sourcePath)
    gtPath = os.listdir(GTPath)
    imNum = len(imgPath)
    mor = 0
    for i in range(imNum):
        img = cv2.imread(sourcePath + imgPath[i])
        # print(sourcePath + imgPath[i])
        gt = cv2.imread(GTPath + gtPath[i])
        # cv2.imshow("22",gt)
        # cv2.waitKey(0)
        imgNp = np.array(img)
        gtNp = np.array(gt)
        diff = (imgNp - gtNp)
        diff = np.where(diff < 0, -diff, diff)
        tp = np.multiply(imgNp/255,gtNp/255)
        mor = mor + np.sum(tp)/(np.sum(diff/255) + np.sum(tp))
        # print('i',i,'++mor == ', mor)
        # mae = mae + (1.0 / np.size(diff)) * (np.sum(diff / 255))
    mor = (mor / imNum)
    print('mor == ',mor)
    return mor


def txt(txt_path, txt_name, precise_,  recall_, F_,mAE):
    # txt_path = 'E:/Project/Saliency_Matlab/debug/result_mask/test_result/pr/'
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    with open(txt_path + txt_name, 'w') as fwrite:
        fwrite.write('single\n precise = {} \n recall = {}\n '
                     'F= {} \n MAE = {}'.format(precise_, recall_, F_ , mAE))


if __name__ == '__main__':

    # dataset = 'PASCAL'
    # datasetList = ['DUT-OMRON','PASCAL']
    datasetList = ['SOD','DUT', 'ECSSD','HKU-IS','PASCALS']
    # datasetList = ['ECSSD','PASCAL', 'MSRA1K']
    for dataset in datasetList:
        path_bsca = "E:/dataset/4test/"+dataset +"/methodSelect/BSCA/"
        path_drfi = "E:/dataset/4test/" + dataset + "/methodSelect/DRFI/"
        # path_dsr = "E:/dataset/4test/" + dataset + "/methodSelect/DSR/"
        path_rbd = "E:/dataset/4test/"+dataset +"/methodSelect/RBD/"
        path_mask = "E:/dataset/4test/"+dataset +"/gtSelect/"

        # TEST = 'E:/dataset/4test/DUT/methodSelect/TEST/'
        rename(path_bsca)
        txt_path = 'E:/dataset/4test/result/'
        # if os.exists(txt_path):


        # BSCA
        preciseBSCA_, recallBSCA_, FBSCA_ = prSingle(path_bsca, path_mask)
        BSCAmAE = MAE(path_bsca, path_mask)
        txt_name = dataset + 'BSCA.txt'
        txt(txt_path, txt_name, preciseBSCA_,recallBSCA_,FBSCA_,BSCAmAE)

        # DRFI
        preciseDRFI_, recallDRFI_, FDRFI_ = prSingle(path_drfi, path_mask)
        DRFImAE = MAE(path_drfi, path_mask)
        txt_name = dataset + 'DRFI.txt'
        txt(txt_path, txt_name, preciseDRFI_,recallDRFI_,FDRFI_,DRFImAE)

        # # DSR
        # preciseDSR_, recallDSR_, FDSR_ = prSingle(path_dsr, path_mask)
        # DSRmAE = MAE(path_dsr, path_mask)
        # txt_name = dataset + 'DSR.txt'
        # txt(txt_path, txt_name, preciseDSR_,recallDSR_,FDSR_,DSRmAE)

        # rbd
        preciseRBD_, recallRBD_, FRBD_ = prSingle(path_rbd, path_mask)
        RBDmAE = MAE(path_rbd, path_mask)
        txt_name = dataset + 'rbd.txt'
        txt(txt_path, txt_name, preciseRBD_,recallRBD_,FRBD_,RBDmAE)

