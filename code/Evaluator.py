import numpy as np
import sys


class Evaluator:
    @staticmethod  # 静态方法
    def calIou(boxA, boxB):
        '''计算IoU,传入的是xmin,ymin,xmax,ymax'''
        W = max(0, (min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])))
        H = max(0, (min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])))
        inter = W * H
        sa = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        sb = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = sa + sb - inter
        iou = inter / union
        return iou

    @staticmethod
    def getPascalVOCMetrics(cfg, classes, detBoxes, gtBoxes, numPos):
        '''
        通过类别作为关键字，得到每个类别的预测，标签，标签总数
        detBoxes:[class,left,top,right,bottom,score,nameImage]
        gtBoxes:[class,left,top,right,bottom,0]  最后一位0代表没有被标记，1代表已经被标记
        numPos:[class,num]每个类别标签的数量
        '''
        ret = []
        for c in classes:
            det_boxes = detBoxes[c]
            gt_boxes = gtBoxes[c]
            n_pos = numPos[c]
            # 利用得分作为关键字，对预测结果框进行排序
            det_boxes = sorted(det_boxes, key=lambda conf: conf[4], reverse=True)
            # 设置两个与预选框长度相同的列表，分别为TP,FP
            TP = np.zeros(len(det_boxes))  # 预选框中TP个数
            FP = np.zeros(len(det_boxes))  # 预测框中FP个数
            for d in range(len(det_boxes)):
                # 将iou设置为最低
                iouMax = sys.float_info.min
                # 遍历与预测框同一类别gt_boxes
                if det_boxes[d][-1] in gt_boxes:
                    #记录iou：（当前标签框和ground true）最大的位置
                    for j in range(len(gt_boxes[det_boxes[d][-1]])):
                        i = Evaluator.calIou(det_boxes[d][:4], gt_boxes[det_boxes[d][-1]][j][:4])
                        if i > iouMax:
                            iouMax = i
                            jmax = j
                    # 若iouMax>=阈值，并且没有被匹配过，计TP
                    if iouMax >= cfg['iouThreshold']:
                        if gt_boxes[det_boxes[d][-1]][jmax][4]== 0: #表示最大值没有更新
                            TP[d] = 1
                            gt_boxes[det_boxes[d][-1]][jmax][4] = 1
                        # 若之前已经匹配过
                        else:
                             FP[d] = 1
                    # 若是iouMax<阈值，计FP
                    else:
                        FP[d] = 1
                # 若没有对应的类别
                else:
                    TP[d] = 1
            # 计算累计的TP,FP
            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum(FP)
            recall = acc_TP / n_pos
            precison = np.divide(acc_TP, (acc_TP + acc_FP))
            # 利用recall,precison进一步求得AP
            [ap, mprec, mrec, ii] = Evaluator.calculateAveragePrecision(recall, precison)
            r = {
                'class': c,
                'precision': precison,
                'recall': recall,
                'AP': ap,
                'interpolated precision': mprec,
                'interpolated recall': mrec,
                'total positives': n_pos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
            }
            ret.append(r)
        return ret, classes

    @staticmethod
    def calculateAveragePrecision(recall, precison):
        '''根据recall和precison计算AP,AP是按面积算的'''
        # recall为x，从0到1
        mrec = []
        mrec.append(0)
        [mrec.append(i) for i in recall]
        mrec.append(1)

        mpre = []
        mpre.append(0)
        [mpre.append(i) for i in precison]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ii = []  # 用来存放x不相同的位置，x相同，y不同会产生二义性
        for i in range(len(mrec) - 1):
            if mrec[i] != mrec[i + 1]:
                ii.append((i + 1))
        ap = 0
        # recall为x轴，precison为y轴，一小块区域
        for index in ii:
            ap = ap + np.sum((mrec[index] - mrec[index - 1]) * mpre[index])
        return [ap, mpre[0:(len(mpre) - 1)], mrec[0:(len(mpre) - 1)], ii]
