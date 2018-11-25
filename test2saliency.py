import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# 修改文件名
def remove(path,suffix):
    # path = "E:/Dataset/img/DUT-OMRON/"
    for infile in glob.glob(os.path.join(path, '*.' + suffix)):
        os.remove(infile)

def rename(path):
    # path = "E:/Project/Saliency_Matlab/debug/result_mask/test_res/GMR/"
    fileList = os.listdir(path)
    counter = 0
    for item in fileList:
        oldname = path + fileList[counter]
        newname = path + item[0:4] + ".png"
        counter +=1
        os.rename(oldname,newname)
        print(newname)

def pr(img_path, mask_path):
    precise = list()
    recall = list()
    # precise = range(256)
    # recall = range(256)
    fileList_img = os.listdir(img_path)
    fileList_mask = os.listdir(mask_path)
    num = len(fileList_mask) - 1

    for j in range(256):
    # for j in range(1):
        sum_fp_tp = 0
        sum_fn_tp = 0
        tp = 0
        for i in range(len(fileList_mask)):
        # for i in range(len(fileList_mask)):
            # print("num = :", len(fileList_mask))
            filePath = os.path.join(img_path, fileList_img[i])
            if os.path.isfile(filePath):
                # print(filePath)
                pic = cv2.imread(filePath, 0)
                imgNp = np.array(pic)
                imgNp = np.where((imgNp) >= j, 255, 0)
                sum_fp_tp = np.sum(imgNp/255)
            filePath_mask = os.path.join(mask_path, fileList_mask[i])
            if os.path.isfile(filePath_mask):
                pic_mask = cv2.imread(filePath_mask, 0)
                maskNp = np.array(pic_mask)
                sum_fn_tp = np.sum(maskNp/255)
                tp_test = np.multiply(imgNp/255, maskNp/255)
                tp = np.sum(tp_test)
                precise.append(tp / sum_fp_tp)
                recall.append(tp / sum_fn_tp)
        print(img_path)
        print("num = ",j, "final precise =  ", precise[j],"final recall == ", recall[j])
    return precise, recall

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

def PRCurve(sourcePath, GTPath):
    imgSourceFiles = os.listdir(sourcePath)
   # imgSourceFiles.sort()
    GTSourceFiles = os.listdir(GTPath)
  # GTSourceFiles.sort()
    imgNumber = len(imgSourceFiles)
    Precision = np.zeros(256, dtype=np.float64)
    Recall = np.zeros(256, dtype = np.float64)
    for i in range(256):
        for j in range(imgNumber):
            img = cv2.imread(sourcePath + imgSourceFiles[j])
            imgGT = cv2.imread(GTPath +GTSourceFiles[j])
            imgArray = np.array(img)
            imgGTArray = np.array(imgGT)
            imgArray = ((imgArray) >= i) * 1
            imgGTArray = ((imgGTArray) > 0) * 2
            diff = imgArray - imgGTArray
            TP = (np.sum(diff == -1))
            FP = (np.sum(diff == 1))
            FN = (np.sum(diff == -2))
            Precision[i] = Precision[i] + (TP * 1.0/(TP + FP ))
            # Precision[i] = Precision[i] + (TP * 1.0/(TP + FP + 1))
            # Recall[i] = Recall[i] + (TP * 1.0/(TP + FN + 1))
            Recall[i] = Recall[i] + (TP * 1.0/(TP + FN ))
        Precision[i] = Precision[i]/imgNumber
        Recall[i] = Recall[i]/imgNumber
        print("complete: ", i, "total: ", 255)
    return Precision, Recall

def F_Measure(sourcePath, GTPath):
    imgSourceFiles = os.listdir(sourcePath)
    # imgSourceFiles.sort()
    GTSourceFiles = os.listdir(GTPath)
    # GTSourceFiles.sort()
    imgNumber = len(imgSourceFiles)
    beta = 0.3
    Precision = 0
    Recall = 0
    for i in range(imgNumber):
        img = cv2.imread(sourcePath + imgSourceFiles[i])
        # print(sourcePath + imgSourceFiles[i])
        # print(GTPath +GTSourceFiles[i])
        imgGT = cv2.imread(GTPath +GTSourceFiles[i])
        imgArray = np.array(img)
        imgGTArray = np.array(imgGT)
        threhold = 2 * np.mean(imgArray)
        imgArray = ((imgArray) > threhold) * 1
        imgGTArray = ((imgGTArray) > 0) * 2
        diff = imgArray - imgGTArray
        TP = (np.sum(diff == -1))
        FP = (np.sum(diff == 1))
        FN = (np.sum(diff == -2))
        Precision = Precision + (TP * 1.0/(TP + FP + 1))
        Recall = Recall + (TP * 1.0/(TP + FN + 1))
        print("complete: ",i, "total: ", imgNumber)
    Precision = Precision*1.0/imgNumber
    Recall = Recall*1.0/imgNumber
    F = ((1+beta*beta)*Precision * Recall)/(beta*beta*Precision+Recall)
    return Precision, Recall, F

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

def histMAE(savePath,mae,maeSingle,name):
    plt.xlabel(name)
    plt.ylabel(u"MAE(mean average error)")
    plt.xticks((0, 1, 2), (u"MINE", u"RBD", u"gtSingle"))
    name_list = ['PSD']
    # name_list = ['SRD','SMD','RC','GS','RBD']
    num_list = mae
    num_list1 = maeSingle
    x = list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, num_list, width=width, label='multi', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='single', fc='r')
    plt.bar(x, num_list1, width=width, label='single', tick_label=name_list, fc='r')
    plt.legend()
    name = name + '_MAESingle2lab.png'
    plt.savefig(savePath + name)
    plt.show()

def histMOR(savePath,mor,morSingle,name):
    plt.xlabel(name)
    plt.ylabel(u"MOR")
    plt.xticks((0, 1), (u"RBD", u"gtSingle"))
    name_list = ['SRD','SMD','RC','GS','RBD']
    num_list = mor
    num_list1 = morSingle
    x = list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, num_list, width=width, label='multi', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='single',  fc='r')
    plt.bar(x, num_list1, width=width, label='single', tick_label=name_list, fc='r')
    plt.legend()
    name = name + '_MORSingle2lab.png'
    plt.savefig(savePath + name)
    plt.show()

def histF(savePath,F,FSingle,name):
    # plt.xlabel(name)
    plt.ylabel(u"F")
    plt.xticks((0, 1), (u"RBD", u"gtSingle"))
    name_list = ['SRD','SMD','RC','GS','RBD']
    num_list = F
    num_list1 = FSingle
    x = list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, num_list, width=width, label='multi', fc='y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='single', tick_label=name_list, fc='r')
    plt.legend()
    name = name + '_FS  ingle2lab.png'
    plt.savefig(savePath + name)
    plt.show()

# def plotPR(savePath,x1,y1,x2,y2):
def plotPR(savePath,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10):
    # 设定范围
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.plot(x1,y1,label='SRD')
    plt.plot(x2,y2,label='SRDSingle')
    plt.plot(x3,y3,label='SMD')
    plt.plot(x4,y4,label='SMDSingle')
    plt.plot(x5,y5,label='RC')
    plt.plot(x6,y6,label='RCSingle')
    plt.plot(x7,y7,label='GS')
    plt.plot(x8,y8,label='GSSingle')
    plt.plot(x9,y9,label='RBD')
    plt.plot(x10,y10,label='RBDSingle')
    plt.legend()
#   保存
    name = 'testPR.png'
    plt.savefig(savePath+name)
    plt.show()

# def plotF(savePath,y1,y2):
def plotF(savePath,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10):
    # 设定范围
    plt.xlim(0,255)
    plt.ylim(0,1)
    x= np.linspace(0,255,256)
    # print(len(x))
    plt.plot(x,y1,label='SRD')
    plt.plot(x,y2,label='SRDSINGLE')
    plt.plot(x,y3,label='SMD')
    plt.plot(x,y4,label='SMDSingle')
    plt.plot(x,y5,label='RC')
    plt.plot(x,y6,label='RCSingle')
    plt.plot(x,y7,label='GS')
    plt.plot(x,y8,label='GSSingle')
    plt.plot(x,y9,label='RBD')
    plt.plot(x,y10,label='RBDSingle')
    plt.legend()
#   保存
    name = 'testF.png'
    plt.savefig(savePath+name)
    plt.show()

def txt(txt_path, txt_name, precise_,  recall_, F_):
    # txt_path = 'E:/Project/Saliency_Matlab/debug/result_mask/test_result/pr/'
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    with open(txt_path + txt_name, 'w') as fwrite:
        fwrite.write('single\n precise = {} \n recall = {}\n '
                     'F= {}'.format(precise_, recall_, F_))


if __name__ == '__main__':

    # dataset = 'PASCAL'
    # datasetList = ['DUT-OMRON','PASCAL']
    datasetList = ['ECSSD','PASCAL', 'MSRA1K','DUT-OMRON']
    # datasetList = ['ECSSD','PASCAL', 'MSRA1K']
    for dataset in datasetList:
        # path_srd = "E:/Dataset/result/"+dataset +"/SRD/"
        # path_srdSingle = "E:/Dataset/result/single2Lab/"+dataset +"/SMD/"
        # path_smd = "E:/Dataset/result/" + dataset + "/SMD/"
        # path_smdSingle = "E:/Dataset/result/single2Lab/" + dataset + "/SMD/"
        # path_rc = "E:/Dataset/result/" + dataset + "/RC/"
        # path_rcSingle = "E:/Dataset/result/single2Lab/" + dataset + "/RC/"
        # path_gs = "E:/Dataset/result/"+dataset +"/GS/"
        # path_gsSingle = "E:/Dataset/result/single2Lab/"+dataset +"/GS/"
        # path_rbd = "E:/Dataset/result/"+dataset +"/RBD/"
        # path_rbdSingle = "E:/Dataset/result/single2Lab/"+dataset +"/RBD/"
        # path_mask = "E:/Dataset/gt/"+dataset +"/"
        txt_path = 'E:/Dataset/result/single2Lab/' + dataset + '/DL/'
        plot_path = 'E:/Dataset/result/single2Lab/' + dataset + '/DL/'
        # rename(path_test)

        # preciseTest_, recallTest_ = prSingle(path_test, path_mask_)
        # FTest_ = F(preciseTest_, recallTest_)
        # path_srd = "E:/Dataset/img/testE/result/rbd/"
        # path_srdSingle = "E:/Dataset/img/testE/result/singleRBD/"
        # path_mask = "E:/Dataset/img/testE/gt/"
        path_srd = "E:/Dataset/result/single2Lab/ECSSD/pubcode/"
        path_srdSingle = "E:/Dataset/result/single2Lab/ECSSD/pubcode1/"
        path_mask = "E:/Dataset/gt/ECSSD/"

        mor = list()
        morItem = MOR(path_srd, path_mask)
        mor.append(morItem)
        # morItem = MOR(path_smd, path_mask)
        # mor.append(morItem)
        # morItem = MOR(path_rc, path_mask)
        # mor.append(morItem)
        # morItem = MOR(path_gs, path_mask)
        # mor.append(morItem)
        # morItem = MOR(path_rbd, path_mask)
        # mor.append(morItem)

        morSingle = list()
        morItem = MOR(path_srdSingle, path_mask)
        morSingle.append(morItem)
        # morItem = MOR(path_smdSingle, path_mask)
        # morSingle.append(morItem)
        # morItem = MOR(path_rcSingle, path_mask)
        # morSingle.append(morItem)
        # morItem = MOR(path_gsSingle, path_mask)
        # morSingle.append(morItem)
        # morItem = MOR(path_rbdSingle, path_mask)
        # morSingle.append(morItem)
        # histMOR(plot_path, mor, morSingle, dataset)

        mae = list()
        maeItem = MAE(path_srd, path_mask)
        mae.append(maeItem)
        # maeItem = MAE(path_smd, path_mask)
        # mae.append(maeItem)
        # maeItem = MAE(path_rc, path_mask)
        # mae.append(maeItem)
        # maeItem = MAE(path_gs, path_mask)
        # mae.append(maeItem)
        # maeItem = MAE(path_rbd, path_mask)
        # mae.append(maeItem)

        maeSingle = list()
        maeItem = MAE(path_srdSingle, path_mask)
        maeSingle.append(maeItem)
        # maeItem = MAE(path_smdSingle, path_mask)
        # maeSingle.append(maeItem)
        # maeItem = MAE(path_rcSingle, path_mask)
        # maeSingle.append(maeItem)
        # maeItem = MAE(path_gsSingle, path_mask)
        # maeSingle.append(maeItem)
        # maeItem = MAE(path_rbdSingle, path_mask)
        # maeSingle.append(maeItem)

        histMAE(plot_path,mae,maeSingle,dataset)



        # SRD
        preciseSRD_, recallSRD_, FSRD_ = prSingle(path_srd, path_mask)
        # preciseSRD, recallSRD = PRCurve(path_srd, path_mask)
        # FSRD = F(preciseSRD, recallSRD)
        txt_name = 'SRD.txt'
        txt(txt_path, txt_name, preciseSRD_,recallSRD_,FSRD_)
        # txt(txt_path, txt_name, preciseSRD, preciseSRD_, recallSRD, recallSRD_, FSRD, FSRD_)

        preciseSRDSingle_, recallSRDSingle_, FSRDSingle_ = prSingle(path_srdSingle, path_mask)
        # preciseSRDSingle, recallSRDSingle = PRCurve(path_srdSingle, path_mask)
        # FSRDSingle = F(preciseSRDSingle, recallSRDSingle)
        txt_name = 'SRDsingle.txt'
        txt(txt_path, txt_name,preciseSRDSingle_, recallSRDSingle_, FSRDSingle_)
        # txt(txt_path, txt_name, preciseSRDSingle ,preciseSRDSingle_, recallSRDSingle ,recallSRDSingle_,FSRDSingle, FSRDSingle_)

        # SMD
        preciseSMD_, recallSMD_, FSMD_ = prSingle(path_smd, path_mask)
        # preciseSMD, recallSMD = PRCurve(path_smd, path_mask)
        # FSMD = F(preciseSMD, recallSMD)
        txt_name = 'SMD.txt'
        txt(txt_path, txt_name, preciseSMD_, recallSMD_, FSMD_)

        preciseSMDSingle_, recallSMDSingle_, FSMDSingle_ = prSingle(path_smdSingle, path_mask)
        # preciseSMDSingle, recallSMDSingle = PRCurve(path_smdSingle, path_mask)
        # FSMDSingle = F(preciseSMDSingle, recallSMDSingle)
        txt_name = 'SMDsingle.txt'
        txt(txt_path, txt_name ,preciseSMDSingle_ ,recallSMDSingle_, FSMDSingle_)

        # RC
        preciseRC_, recallRC_, FRC_ = prSingle(path_rc, path_mask)
        # preciseRC, recallRC = PRCurve(path_rc, path_mask)
        # FRC = F(preciseRC, recallRC)
        txt_name = 'RC.txt'
        txt(txt_path, txt_name, preciseRC_, recallRC_, FRC_)

        preciseRCSingle_, recallRCSingle_, FRCSingle_ = prSingle(path_rcSingle, path_mask)
        # preciseRCSingle, recallRCSingle = PRCurve(path_rcSingle, path_mask)
        # FRCSingle = F(preciseRCSingle, recallRCSingle)
        txt_name = 'RCsingle.txt'
        txt(txt_path, txt_name ,preciseRCSingle_ ,recallRCSingle_, FRCSingle_)

        # GS
        preciseGS_, recallGS_, FGS_ = prSingle(path_gs, path_mask)
        # preciseGS, recallGS = PRCurve(path_gs, path_mask)
        # FGS = F(preciseGS, recallGS)
        txt_name = 'GS.txt'
        txt(txt_path, txt_name, preciseGS_, recallGS_, FGS_)

        preciseGSSingle_, recallGSSingle_, FGSSingle_ = prSingle(path_gsSingle, path_mask)
        # preciseGSSingle, recallGSSingle = PRCurve(path_gsSingle, path_mask)
        # FGSSingle = F(preciseGSSingle, recallGSSingle)
        txt_name = 'GSsingle.txt'
        txt(txt_path, txt_name ,preciseGSSingle_ ,recallGSSingle_, FGSSingle_)

        # RBD
        preciseRBD_, recallRBD_, FRBD_ = prSingle(path_rbd, path_mask)
        # preciseRBD, recallRBD = PRCurve(path_rbd, path_mask)
        # FRBD = F(preciseRBD, recallRBD)
        txt_name = 'RBD.txt'
        txt(txt_path, txt_name, preciseRBD_, recallRBD_, FRBD_)

        preciseRBDSingle_, recallRBDSingle_, FRBDSingle_ = prSingle(path_rbdSingle, path_mask)
        # preciseRBDSingle, recallRBDSingle = PRCurve(path_rbdSingle, path_mask)
        # FRBDSingle = F(preciseRBDSingle, recallRBDSingle)
        txt_name = 'RBDsingle.txt'
        txt(txt_path, txt_name ,preciseRBDSingle_ ,recallRBDSingle_, FRBDSingle_)

        #
        F = list()
        FSingle = list()
        F.append(FSRD_)
        FSingle.append(FSRDSingle_)
        F.append(FSMD_)
        FSingle.append(FSMDSingle_)
        F.append(FRC_)
        FSingle.append(FRCSingle_)
        F.append(FGS_)
        FSingle.append(FGSSingle_)
        F.append(FRBD_)
        FSingle.append(FRBDSingle_)
        histF(plot_path,F,FSingle,dataset)

        # plotF(plot_path,FSRD, FSRDSingle,FSMD, FSMDSingle,FRC, FRCSingle,FGS, FGSSingle,FRBD, FRBDSingle)
        # plotPR(plot_path,recallSRD, preciseSRD, recallSRDSingle, preciseSRDSingle,recallSMD, preciseSMD, recallSMDSingle, preciseSMDSingle
        #        , recallRC, preciseRC, recallRCSingle, preciseRCSingle,recallGS, preciseGS, recallGSSingle, preciseGSSingle,recallRBD, preciseRBD, recallRBDSingle, preciseRBDSingle)
