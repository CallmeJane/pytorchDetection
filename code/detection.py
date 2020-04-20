import os
from Evaluator import *
import matplotlib.pyplot as plt


def getGTBoxes(cfg, gtFolder):
    '''
    GTBoxes:{class,[left,top,right,bottom,0/1]},最后一位0表示未访问过，1表示访问过
    classes:[]每个类别名称
    num_pos:{}每个类别的个数
    '''
    files = os.listdir(gtFolder)
    files.sort()
    gt_boxes = {}
    classes = []  # 每个类别名称
    num_pos = {}  # 每个类别的个数
    for f in files:
        nameOfImage = f.replace('.txt', '')
        strs = open(os.path.join(gtFolder, f), 'r')
        for line in strs:
            line = line.replace('\n', '')  # 需要重新赋值
            if line.replace(' ', '') == '':
                continue
            spiltSpace = line.split(' ')
            className = (spiltSpace[0])
            left = float(spiltSpace[1])
            top = float(spiltSpace[2])
            right = float(spiltSpace[3])
            bottom = float(spiltSpace[4])
            one_box = [left, top, right, bottom, 0]
            if className not in classes:
                classes.append(className)
                gt_boxes[className] = {}
                num_pos[className] = 0
            num_pos[className] += 1
            if nameOfImage not in gt_boxes[className]:
                gt_boxes[className][nameOfImage] = []
            gt_boxes[className][nameOfImage].append(one_box)
        strs.close()
    return gt_boxes, classes, num_pos


def getDetBoxes(cfg, detFolder):
    '''
    DetBoxes {className,[left,top,right,bottom,score,NameofImage]}
    '''
    det_boxes = {}
    files = os.listdir(detFolder)
    for f in files:
        nameofImage = f.replace('.txt', '')
        fstr = open(os.path.join(detFolder, f), 'r')
        for line in fstr:
            line = line.replace('\n', '')
            if line.replace(' ', '') == '':
                continue
            splitSpace = line.split(" ")
            cls = (splitSpace[0])
            left = float(splitSpace[1])
            top = float(splitSpace[2])
            right = float(splitSpace[3])
            bottom = float(splitSpace[4])
            score = float(splitSpace[5])
            one_box = [left, top, right, bottom, score, nameofImage]
            if cls not in det_boxes:
                det_boxes[cls] = []
            det_boxes[cls].append(one_box)
        fstr.close()
    return det_boxes


def detections(cfg, gtFolder, defFolder, save_path):
    gt_boxes, classes, num_pos = getGTBoxes(cfg, gtFolder)
    detBoxes = getDetBoxes(cfg, defFolder)
    ret, classes = Evaluator.getPascalVOCMetrics(cfg, classes, detBoxes, gt_boxes, num_pos)
    return ret, classes


def calmAPAndPlot(cfg, results, classes, save_path):
    plt.rcParams['savefig.dpi'] = 80  # 图片像素
    plt.rcParams['figure.dpi'] = 130  # 分辨率

    acc_AP = 0
    figure_index = 0
    for clc_index, result in enumerate(results):
        if result == None:
            raise IOError('%d can not be found', clc_index)
        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']
        figure_index += 1

        plt.figure(figure_index)
        plt.plot(recall, precision, cfg['colors'][clc_index], label='Precison')
        plt.xlabel('recall')
        plt.ylabel('precison')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(shadow='True')
        plt.grid()
        plt.savefig(os.path.join(save_path, cls + '.png'))
        plt.show()
        plt.pause(0.05)
    mAP = acc_AP / figure_index
    mAP_str = '{0:.2f}%'.format(mAP * 100)
    print('mAP=%f' % (mAP))
