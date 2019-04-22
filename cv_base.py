
import os
import os.path as ops
import cv2
import numpy as np
import yaml
import rawpy
from PIL import Image
import sys
print(__file__)
sys.path.append(ops.dirname(__file__)) 
import matplotlib.pyplot as plt
import math
from utils import *
from tree import Tree

import random
'''
本文件当中存放图像处理常用工具。
我会尽量在中间增加说明的。
这个文件当中存储一些静态方法。
当要使函数接收元组或字典形式的参数 的时候，有一种特殊的方法，它分别使用*和**前缀 。这种方法在函数需要获取可变数量的参数的时候特别有用。
[注意] 
[1] 由于在args变量前有*前缀 ，所有多余的函数参数都会作为一个元组存储在args中 。如果使用的是**前缀 ，多余的参数则会被认为是一个字典的健/值对 。
[2] 对于def func(*args):，*args表示把传进来的位置参数存储在tuple（元组）args里面。例如，调用func(1, 2, 3) ，args就表示(1, 2, 3)这个元组 。
[3] 对于def func(**args):，**args表示把参数作为字典的健-值对存储在dict（字典）args里面。例如，调用func(a='I', b='am', c='wcdj') ，args就表示{'a':'I', 'b':'am', 'c':'wcdj'}这个字典 。
[4] 注意普通参数与*和**参数公用的情况，一般将*和**参数放在参数列表最后。
'''

class CVBase(object):
    '''
    构造函数，基本没啥用吧qwq，因为传递的东西都是图片。
    '''
    def __init__(self, filename, config):
        '''
        处理的标签序号
        '''
        self.index = 0
        '''
        正在执行的方法，从yaml文件当中读取
        '''
        self.methods = []
        '''
        有不同标签的图片存放在这里
        '''
        self.labels = []
        '''
        中间结果，有个直方图的结果放在这里
        '''
        self.hists = []
        '''
        正在处理的文件名
        '''
        self.filename = filename
        '''
        配置文件的路径
        '''
        self.params = []
        self.config = config
        self.parse(self.config)

    '''
    从参数文件当中读取一些数据，放在methods里面
    '''
    def parse(self, filename):
        f = open(filename, 'r')
        self.methods = yaml.load(f, Loader=yaml.FullLoader)
        dump(self.methods)

    @staticmethod
    def get_param(params, val, default):
        if val in params:
            return params[val]
        else:
            return default

    '''
    读入文件，
    输入是文件夹
    输出参数是什么full_path的list
    '''
    @staticmethod
    def read_files(filename, filter=None):
        files = os.listdir(filename)
        files_arr=[]
        dirname = ops.abspath(filename)
        # dump(dirname)
        for file_name in files:
            full_name = ops.join(dirname,file_name)
            if ops.isfile(full_name):
                if filter == None or full_name.split('.')[-1] in filter:
                    files_arr.append(full_name)
        files_arr.sort()
        return files_arr

    def _base_verbose(self,params,other):
        res = ''
        res = '第%d张图片，'%(self.index,)
        if exist(params,'input_label'):
            input_label = get(params,'input_label')
            res = res +"输入标签为%s，" % (input_label,)
        res = res+ other
        if exist(params,'output_label'):
            output_label = get(params,'output_label')
            res = res +"输出标签为%s，" % (output_label,)
        return res

    def _readIn(self, params):
        mode = get(params,'mode', -1)
        raw_size = get(params, 'raw_size', 3072)
        if test(params, 'type' ,'raw'):
            image_data =  open(self.file,'rb').read()
            im = Image.frombytes('L',(raw_size,raw_size*2),image_data,'raw')
            np_im = np.array(im)
            new_im = np_im[::2,:]
            new_im_2 = np_im[1::2,:]
            dump(new_im.shape)
            img = Image.fromarray(new_im.astype('uint8'))
            img = img.resize((raw_size/2,raw_size), Image.BOX)
            img2 = Image.fromarray(new_im_2.astype('uint8'))
            img2 = img2.resize((raw_size/2,raw_size), Image.BOX)
            new_img = Image.new('L',(raw_size,raw_size))
            new_img.paste(img, (0,0))
            new_img.paste(img2, (raw_size/2,0))
            pic = new_img
        else:
            pic = cv2.imread(self.file, flags=mode)
        label = get(params, 'output_label', 'img'+str(self.index))
        self.labels[self.index][label] = pic
        if test(params, 'verbose', True):
            other = '操作为%s，叫做%s的图片读入了，'%(sys._getframe().f_code.co_name,self.basename,)
            res = self._base_verbose(params,other)
            dump(res)

    def _print(self, params):
        if test(params, 'mode', 'shape'):
            if exist(params, 'input_label'):
                label_name = get(params, 'input_label', 'hello')
                test_pic = self.labels[self.index][label_name]
                if test(params, 'verbose', True):
                    other = '操作为%s，这张图片的高是%d，宽是%d,' % (sys._getframe().f_code.co_name, test_pic.shape[0],test_pic.shape[1])
                    res = self._base_verbose(params,other)
                    dump(res)
                else:
                    dump(self.labels[self.index][label_name].shape)

    def _threshold(self, params):
        if exist(params, 'input_label'):
            input_name = get(params, 'input_label', 'hello')
        origin_pic = self.labels[self.index][input_name]
        val = get(params,'val',1)
        if isinstance(val,int):
            pass
        else:
            val = self.params[self.index][param_out[val]]
        ret,pic = cv2.threshold(origin_pic,val,255, cv2.THRESH_BINARY)
        if exist(params, 'output_label'):
            label_name = get(params, 'output_label', 'hello')
            self.labels[self.index][label_name] = pic
        if test(params, 'verbose', True):
            other = '门限参数为%d，' % (val,)
            res = self._base_verbose(params,other)
            dump(res)
    
    '''
    输出图像
    path: 输出路径
    val: 输出的是哪一张图像，
    '''
    def _writeOut(self, params):
        path = ops.abspath(params['path'])
        if exist(params, 'input_label'):
            label_name = get(params, 'input_label', 'hello')
        else:
            raise ValueError(sys._getframe().f_code.co_name+'这个函数没有'+'input_label'+'标签啊')
        if not ops.exists(path):
            os.makedirs(path)
        filename = os.path.basename(self.file)
        prefix = get(params, 'prefix','')
        suffix = get(params,'suffix')
        filename = prefix + filename
        t = filename.split('.')
        t[-1] = suffix + '.' + t[-1]
        filename = ''.join(t)
        out_name = ops.join(path, filename)
        cv2.imwrite(out_name,self.labels[self.index][label_name])
        if test(params, 'verbose', True):
            other = '操作为%s，保存文件名为%s，' % (sys._getframe().f_code.co_name,filename,)
            res = self._base_verbose(params,other)
            dump(res)
    
    def _drawAreaHist(self,params):
        channel = get(params, 'channel', 30)
        pic_width = get(params, 'width', 400)
        pic_height = get(params, 'height', 300)
        param_in = get(params, 'param_in', {})
        param_out = get(params, 'param_out', {})
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        areas = self.params[self.index][param_in['areas']]
        pic = np.zeros([pic_height,pic_width,3], np.uint8)
        color = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
        hist_list = np.zeros(channel)
        minVal = np.min(areas)
        maxVal = np.max(areas)
        for i in range(areas.shape[0]):
            max_err = maxVal - minVal
            err = areas[i] - minVal
            index = int(err * (channel-1) / max_err)
            hist_list[index] += 1
        
        areas_range = np.linspace(minVal,maxVal,channel)
        print('areas_range',areas_range, areas_range.shape)
        self.params[self.index][param_out['areas_range']] = areas_range

        hist_list = np.array(hist_list)
        self.params[self.index][param_out['areas_hist']] = hist_list
        print('hist_list',hist_list, hist_list.shape)
        maxVal = np.max(hist_list)
        for h in range(channel):  
            color = (random.randrange(60,255),random.randrange(60,255),random.randrange(60,255))
            width = int(pic_width/channel)
            height = int(hist_list[h]*0.9*pic_height/maxVal)
            # print((h*width,pic_height-height), (h*width+width,pic_height))
            cv2.rectangle(pic,(h*width,pic_height-height), (h*width+width,pic_height), color,-1) 
        
        one_area = (areas_range[np.argmax(hist_list)] + areas_range[np.argmax(hist_list)+1])/2
        self.params[self.index][param_out['one_area']] = one_area

        output_img = pic
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    def _writeFile(self, params):
        pass

    def _morphology(self,params):
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label].copy()
        kernel_size = get(params, 'kernel_size',5)
        method = get(params, 'method',5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
        for item in params['run']:
            erode_time = get(item,'erode_time',1)
            dilate_time = get(item,'dilate_time',1)
            times = get(item,'times',1)
            temp = input_img
            res = get(item, 'res', 0)
            for i in range(times):
                if erode_time > 0:
                    erode = cv2.dilate(temp, kernel, iterations = erode_time)
                else:
                    erode = temp
                if dilate_time > 0:
                    dilate = cv2.erode(temp, kernel, iterations = dilate_time)
                else:
                    dilate = temp
                if res == 0:
                    temp = erode - dilate
                elif res == 1:
                    temp = dilate - erode
                elif res == 2:
                    temp = erode + dilate
                elif res == 3:
                    temp = - erode - dilate
            input_img = temp
        output_img = temp
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        if test(params, 'verbose',True):
            other = '操作为%s，形态学核大小为%d，' % (sys._getframe().f_code.co_name,kernel_size)
            res = self._base_verbose(params,other)
            dump(res)

    def _getOneArea(self, params):
        param_out = get(params, 'param_out',{})
        param_in = get(params, 'param_in',{})       
        max_length = get(params,'max_length', 1000)
        min_length = get(params, 'min_length',7)
        min_one_size = get(params,'min_one_size',45) 
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        # 找轮廓
        contours, hierarchy = cv2.findContours(input_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        areas = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            # 检验宽度和高度，如果小于10或者大于1000的话跳过
            if rect[1][0] >max_length or rect[1][1]> max_length or rect[1][0] < min_length or rect[1][1] <min_length:
                continue
            # 检验面积
            rect_area=rect[1][0]*rect[1][1]
            if rect_area < min_one_size:             # 细小的轮廓跳过
                continue

            area = cv2.contourArea(contour,oriented=True)
            if area < 0:
                continue
            if area > 50000:
                continue

            # 在这里统计每个轮廓的面积
            areas.append(area)
        
        # areas = np.sort(np.array(areas))
        areas = np.array(areas)
        # print(areas)
        # one_area = np.mean(areas[100:200])
        # print(one_area, '*********************')
        self.params[self.index][param_out['areas']] = areas
        # self.params[self.index][param_out['one_area']] = one_area

        output_img = input_img
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')
    
    def _countByArea(self, params):
        param_in = get(params, 'param_in')
        param_out = get(params, 'param_out')
        max_length = get(params,'max_length', 1000) 
        min_length = get(params, 'min_length',7)
        min_one_size = get(params,'min_one_size',45)
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作        
        contours, hierarchy = cv2.findContours(input_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if isinstance(param_in['one_area'],int):
            one_area = param_in['one_area']
        else:
            one_area = self.params[self.index][param_in['one_area']]
        print(one_area,'*****************')
        count = 0
        output_img = input_img.copy()
        font=cv2.FONT_HERSHEY_SIMPLEX
        output_img = cv2.cvtColor(input_img,cv2.COLOR_GRAY2BGR)
        for contour in contours:
            color = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
            rect = cv2.minAreaRect(contour)
            # 检验宽度和高度，如果小于10或者大于1000的话跳过
            if rect[1][0] >max_length or rect[1][1]> max_length or rect[1][0] < min_length or rect[1][1] <min_length:
                continue
            # 检验面积
            area=rect[1][0]*rect[1][1]
            if area < min_one_size:             # 细小的轮廓跳过
                continue
            area = cv2.contourArea(contour)
            area_count = int(round(area / one_area))
            count = count + area_count
            cv2.drawContours(output_img,[contour],0,color,2)
            cv2.putText(output_img,str(area_count),(int(rect[0][0]),int(rect[0][1])), font, 1,color,1)
        
        self.params[self.index][param_out['area_count']] = count
        # 这里已经得到了直方图，如何选择one_area

        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，统计数量为%s，' % (sys._getframe().f_code.co_name,count)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')


    '''
    画颜色直方图，这个一般是黑白图，注意二值图不行，一般是拿这个结果，找到二值图的阈值的。
    '''
    def _drawColorHist(self, params):
        histImg = np.zeros([256,256,3], np.uint8)
        hist_list = []
        color = (255,255,255)
        channel = get(params,'channel')
        smooth = get(params, 'smooth')
        threshold = get(params, 'threshold')
        smooth_param = get(params, 'smooth_param',10)
        param_out = get(params, 'param_out',{})
        if exist(params, 'input_label'):
            label_name = get(params, 'input_label', 'hello')
            test_pic = self.labels[self.index][label_name]
        hist = cv2.calcHist([test_pic], [0], None, [channel], [0.0,255.0])
        _ , maxVal, _ , _ = cv2.minMaxLoc(hist)  
        hpt = int(0.9* 256)  
        for h in range(channel):    
            width = int(256.0/channel)
            height = int(hist[h]*hpt/maxVal) 
            hist_list.append(height)
            cv2.rectangle(histImg,(h*width,256-height), (h*width+width,256), color,1)    
            # cv2.rectangle(histImg,(),())
        
        median = np.median( np.array(hist_list) )
        self.params[self.index][param_out['mid_hist']] = median
        
        if threshold > 0 :
            cv2.line(histImg,( int(round(median)) ,256), ( int(round(median)) ,0), [255,255,0]) 
        self.params[self.index][param_out['hist']] = hist_list

        if smooth == True:
            temp=self.params[self.index][param_out['hist']]
            for i in range(smooth_param,len(temp)-smooth_param):
                sum=0
                for j in range(-smooth_param,smooth_param+1):
                    sum=sum+self.params[self.index][param_out['hist']][i+j]
                temp[i]=sum/2.0/smooth_param
            for h in range(channel):    
                width = int(256.0/channel)
                height = int(temp[h]*hpt/maxVal) 
                cv2.rectangle(histImg,(h*width,256-height), (h*width+width,256), color,1)  
        if exist(params, 'output_label'):
            output_name = get(params, 'output_label', 'hello')
            self.labels[self.index][output_name] = histImg
        if test(params, 'verbose',True):
            other = '操作为%s，光滑开关为%d，门限为%d，光滑参数为%d，' % (sys._getframe().f_code.co_name,smooth,threshold, smooth_param)
            res = self._base_verbose(params,other)
            dump(res)
    
    def _findCenterAndContours(self, params):
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        # 读入参数
        max_length = get(params,'max_length', 1000) 
        min_length = get(params, 'min_length',7)
        circle_size = get(params,'circle_size',50000)
        param_out = get(params, 'param_out',{})
        # 找轮廓
        contours, hierarchy = cv2.findContours(input_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        # 找到中心点
        center = list(input_img.shape)
        center[0]/=2
        center[1]/=2
        flag = False
        new_contours = []
        rects = []
        # 第一次循环，找到圆心
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            # 检验宽度和高度，如果小于10或者大于1000的话跳过
            if rect[1][0] >max_length or rect[1][1]> max_length or rect[1][0] < min_length or rect[1][1] <min_length:
                continue
            # 检验面积
            area=rect[1][0]*rect[1][1]
            box = cv2.boxPoints(rect)

            if not flag and area > circle_size and self.__inBox(box, center):
                center_contour = contour
                flag = True
            else:
                contour_area = cv2.contourArea(contour,oriented=True)
                if contour_area < 0:
                    continue
                rects.append(rect)
                new_contours.append(contour)
                
        output_img = input_img
        # 得到了最近距离和最远距离
        self.params[self.index][param_out['center_contour']] = center_contour
        # 找到中心的x和y坐标
        center_rect = cv2.fitEllipse(center_contour)
        self.params[self.index][param_out['center_rect']] = center_rect
        # 所有的轮廓在这个地方
        self.params[self.index][param_out['new_contours']] = new_contours
        # 所有的外接矩形在这个地方
        self.params[self.index][param_out['rects']] = rects
        
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        if test(params, 'verbose',True):
            other = '操作为%s，中心点的坐标为(%f,%f)，宽度为%f，高度为%f，找到的轮廓（除中心外）有%d个，找到的外接矩形（除中心的外）有%d个，' % (sys._getframe().f_code.co_name,center_rect[0][0],center_rect[0][1],center_rect[1][0],center_rect[1][1],len(new_contours),len(rects))
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')
    
    def __inBox(self,box, point):
        A = np.array(list(box[0]))
        B = np.array(list(box[1]))
        C = np.array(list(box[2]))
        D = np.array(list(box[3]))
        P = np.array(point[:2])
        one = np.cross(B-A, P-A)*np.cross(D-C, A-C)
        two = np.cross(C-B, P-B)*np.cross(A-D, P-D)
        return (one>0 and two>0)

    def _saveContours(self, params):
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]

        param_in = get(params,'param_in',{})
        param_out = get(params,'param_out',{})        
        path = ops.abspath(params['path'])
        ##############################
        # TODO 这里填写输入参数
        ##############################
        ##############################
        # TODO 这里填写具体操作
        ##############################
        new_contours = self.params[self.index][param_in['new_contours']]
        contours_img = []
        # white_img = np.zeros_like(input_img)
        for i,contour in enumerate(new_contours):        
            white_img = np.zeros_like(input_img)
            zero_img = cv2.drawContours(white_img,[contour],-1,(255,255,255),-1)
            rect = cv2.minAreaRect(contour)       
            w,h = rect[1][0],rect[1][1]
            box = cv2.boxPoints(rect)
            new_rect = np.array([[0,0],[0,h],[w,h],[w,0]])
            # patch = np.ones((w,h),dtype=np.uint8)
            perspective_transform = cv2.getPerspectiveTransform(box.astype(np.float32),new_rect.astype(np.float32))
            new_pic = cv2.warpPerspective(zero_img,perspective_transform,(int(w),int(h)))
            if w > h:
                new_pic = cv2.rotate(new_pic,cv2.ROTATE_90_COUNTERCLOCKWISE)
            contours_img.append(new_pic)
            area = cv2.contourArea(contour,oriented = True)
            new_name = os.path.join(path,'contour_' + str(self.index) + '_'+str(i) +'_'+str(area) + '.png')
            cv2.imwrite(new_name,new_pic)

        self.params[self.index][param_out['contours_img']] = contours_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')
    
    def _countByDist(self, params):
        param_in = get(params,'param_in',{})
        param_out = get(params,'param_out',{})
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        # 这边得到平均长度
        rects = self.params[self.index][param_in['rects']]
        new_rects = []
        for rect in rects:
            new_rects.append([rect[0][0],rect[0][1],rect[1][0],rect[1][1], rect[2]])
        new_rects = np.array(new_rects)
        print(new_rects.shape)
        print(new_rects)

        new_rects = new_rects[new_rects[:,2].argsort()]

        # print(rects)
        mean_length = np.mean(new_rects[100:200,2:4],axis = 0)
        print(mean_length)
        center_rect = self.params[self.index][param_in['center_rect']]
        new_contours = self.params[self.index][param_in['new_contours']]

        count = 0
        count_res = []
        dist_list = []
        for contour in new_contours:
            min_dist = 80000
            max_dist = 0
            for point in contour:
                point = point.tolist()[0]
                ex = center_rect[0][0] - point[0]
                ey = center_rect[0][1] - point[1]
                dist = math.sqrt(ex**2+ey**2)
                if dist>max_dist:
                    max_dist = dist
                if min_dist>dist:
                    min_dist = dist
            err = max_dist - min_dist
            if test(params,'mode','width'):
                val = mean_length[0]
            elif test(params,'mode','height'):
                val = mean_length[1]
            predict_num = int(np.round(err / val))
            count_res.append(predict_num)
            dist_list.append(err)
            count += predict_num
        output_img = input_img
        self.params[self.index][param_out['dist_count']] = count
        self.params[self.index][param_out['count_res']] = count_res
        self.params[self.index][param_out['dist_list']] = dist_list
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，总的个数为%d，其他轮廓的数量为%d，' % (sys._getframe().f_code.co_name,count,len(count_res))
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    '''
    在findCenterAndContours和countByDist之后才能够执行
    '''
    def _drawContour(self, params):
        param_in = get(params,'param_in',{})
        param_out = get(params,'param_out',{})
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        output_img = cv2.cvtColor(input_img,cv2.COLOR_GRAY2BGR)
        # new_contours = self.params[self.index][param_in['new_contours']]
        rects = self.params[self.index][param_in['rects']]
        count_res = self.params[self.index][param_in['count_res']]
        linewidth = 1
        font=cv2.FONT_HERSHEY_SIMPLEX
        for index,rect in enumerate(rects):
            color = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
            box = cv2.boxPoints(rect)
            for i in range(4):
                cv2.line(output_img,tuple(box[i]), tuple(box[(i+1)%4]), color, linewidth) # 画线
            cv2.putText(output_img,str(count_res[index]),(int(rect[0][0]),int(rect[0][1])), font, 1,color,1)

        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    '''
    把笛卡尔坐标系变成极坐标系
    '''
    def _toPolar(self,params):
        ##############################
        # TODO 这里填写输入参数
        param_in = get(params,'param_in',{})
        shake_times = get(params, 'shake_times', 0)
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        center_point = self.params[self.index][param_in['center_rect']]
        ox,oy = center_point[0]
        x = np.arange(input_img.shape[0])
        y = np.arange(input_img.shape[1])
        ex = x - ox
        ey = y - oy
        ex2 = ex**2
        ey2 = ex**2
        dist = np.sqrt(ex2.reshape(-1,1) + ey2.reshape(1,-1))
        angle = np.arctan2(ex.reshape(-1,1),ey.reshape(1,-1))
        angle_extend = 3072/np.pi 
        angle = angle * angle_extend
        
        norm_dist = dist.astype(np.int16).reshape(-1)
        norm_angle = angle.astype(np.int16)
        norm_angle = (norm_angle - np.min(norm_angle)).reshape(-1)
        norm_x = np.tile(x.reshape(-1,1),(1,y.shape[0])).reshape(-1)
        norm_y = np.tile(y.reshape(1,-1),(x.shape[0],1)).reshape(-1)

        new_pic = np.zeros( (np.max(norm_dist) +1, np.max(norm_angle) +1))
        # new_pic = np.ones( (np.max(norm_dist) +1, np.max(norm_angle) +1))*(-1)
        # 实际的转化就这一步
        new_pic[norm_dist,norm_angle] = input_img[norm_x,norm_y]

        # 调整扇形位置
        # not_change = np.where(new_pic==-1)
        # mask = np.zeros_like(new_pic)
        # mask[not_change] = 1
        # one_mask = np.argsort(mask,axis=1,kind='mergesort').reshape(-1)
        # new_img = np.zeros_like(new_pic)

        # x = np.arange(new_img.shape[0])
        # y = np.arange(new_img.shape[1])
        # new_x = np.tile(x.reshape(-1,1),(1,y.shape[0])).reshape(-1)
        # new_y = np.tile(y.reshape(1,-1),(x.shape[0],1)).reshape(-1)

        # new_img[new_x,new_y] = new_pic[new_x,one_mask]
        new_img = new_pic

        for _ in range(shake_times):
            new_img[1:,:] = np.maximum(new_img[1:,:] , new_img[:-1,:])
            new_img[:,1:] = np.maximum(new_img[:,1:] , new_img[:,:-1])
        output_img = new_img.astype(np.uint8)
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    '''
    把一个图片复制成另一个名称
    '''
    def _copy(self,params):
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        output_img = input_img
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')
    

    '''
    把一个图片复制成另一个名称
    '''
    def _matchContours(self,params):
        param_in = get(params,'param_in',{})
        val = get(params,'val',0.7)
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        contours_img = self.params[self.index][param_in['contours_img']]
        # 随便选的数字，这里应该有算法才对
        templ = contours_img[22]
        padding = 200
        half_padding = 100
        match_template_count = 0
        for i,contour_img in enumerate(contours_img):
            w,h = contour_img.shape
            padding_contour_img = np.zeros((w+padding,h+padding),dtype=np.uint8 )
            padding_contour_img[half_padding:-half_padding,half_padding:-half_padding] = contour_img
            new_w, new_h = padding_contour_img.shape
            tw,th = templ.shape
            if new_w > tw and new_h>th:
                max_val = self.__flip_pic_find_max_val(padding_contour_img, templ, val, i)
                match_template_count += len(max_val)
            else:
                print('图片%d太小了，才%d,%d不足以被模板%d,%d识别，加了padding也不行qwq'%(i,new_w,new_h,tw,th ))
        print('使用模板匹配统计个数：',match_template_count)
        output_img = input_img
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    def __flip_pic_find_max_val(self, padding_contour_img, templ, val, i):
        # 原图
        collection, max_val = self.__get_max_val(padding_contour_img, templ, val)
        if len(max_val)!=0:
            print(i,collection.T,max_val)
            return max_val

        # 旋转180
        padding_contour_img = cv2.rotate(padding_contour_img,cv2.ROTATE_180)
        collection, max_val = self.__get_max_val(padding_contour_img, templ, val)
        if len(max_val)!=0:
            print(i,'fliped 180',collection.T,max_val)
            return max_val

        # 旋转90
        padding_contour_img = cv2.rotate(padding_contour_img,cv2.ROTATE_90_CLOCKWISE)
        collection, max_val = self.__get_max_val(padding_contour_img, templ, val)
        if len(max_val)!=0:
            print(i,'fliped 90',collection.T,max_val)
            return max_val

        # 再旋转180
        padding_contour_img = cv2.rotate(padding_contour_img,cv2.ROTATE_180)
        collection, max_val = self.__get_max_val(padding_contour_img, templ, val)
        if len(max_val)!=0:
            print(i,'fliped -90',collection.T,max_val)
            return max_val
        
        return []

    def __get_max_val(self, padding_contour_img, templ, val):
        res = cv2.matchTemplate(padding_contour_img,templ,cv2.TM_CCOEFF_NORMED)
        loc = np.where(res > val)
        collection = self.__get_loc(loc, res)
        max_val = []
        if len(collection) != 0:
            max_val = res[collection[0,:],collection[1,:]]
        return collection,max_val

    def __get_loc(self, loc, res):
        if len(loc[0])==0:
            return np.array([])
        collection = [(loc[0][0],loc[1][0])]
        for index in range(loc[0].shape[0]):
            loc_y,loc_x = loc[0][index],loc[1][index]
            flag = False
            for item_index in range(len(collection)):
                y,x = collection[item_index]
                if abs(y-loc_y)< 5 or abs(x-loc_x)< 5:
                    flag = True
                    if res[y,x] < res[loc_y,loc_x]:
                        collection[item_index] = (loc_y,loc_x)
            if flag == False: 
                collection.append((loc_y,loc_x))
        collection = np.array(collection).T
        return collection

    '''
    把一个图片复制成另一个名称
    '''
    def _blur(self,params):
        mode= get(params,'mode','normal')
        kernel_size = get(params,'kernel_size', 3)
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        if mode == 'normal':
            output_img = cv2.blur(input_img,kernel_size)
        elif mode == 'median':
            output_img = cv2.medianBlur(input_img,kernel_size)  
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    '''
    使用滤镜
    '''
    def _filter(self,params):
        kernel_type = get(params, 'kernel_type', 'gaussian')
        kernel_size = get(params, 'kernel_size', 3)
        direction = get(params, 'direction', 0)
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        if (kernel_type,kernel_size,direction) == ('scharr',3,0):
            kernel = np.array([[-3,-10,-3],[0,0,0],[3,10,3]])
            output_img = cv2.filter2D(input_img,ddepth=0,kernel = kernel)
        elif (kernel_type,kernel_size,direction) == ('scharr',3,1):
            kernel = np.array([[-3,0,3],[-10,0,10],[-3,0,3]])
            output_img = cv2.filter2D(input_img,ddepth=0,kernel = kernel)
        elif (kernel_type,kernel_size,direction) == ('sobel',3,1):
            # x
            kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            output_img = cv2.filter2D(input_img,ddepth=0,kernel = kernel)
        elif (kernel_type,kernel_size,direction) == ('sobel',3,0):
            # y
            kernel = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
            output_img = cv2.filter2D(input_img,ddepth=0,kernel = kernel)
        elif (kernel_type,kernel_size,direction) == ('sobel–feldman',3,1):
            # x
            kernel = np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
            output_img = cv2.filter2D(input_img,ddepth=0,kernel = kernel)
        elif (kernel_type,kernel_size,direction) == ('sobel–feldman',3,0):
            # y
            kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            output_img = cv2.filter2D(input_img,ddepth=0,kernel = kernel)
        elif (kernel_type,kernel_size,direction) == ('sobel',3,-1):
            kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            output_img1 = cv2.filter2D(input_img,ddepth=0,kernel = kernel_y)
            output_img2 = cv2.filter2D(input_img,ddepth=0,kernel = kernel_x)
            output_img = np.sqrt(output_img1**2 + output_img2 ** 2) * 16
            # output_img = input_img
        output_img = output_img.astype(np.uint8)
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            print(output_img)
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    '''
    一些数学操作
    '''
    def _math(self,params):
        ##############################
        # TODO 这里填写输入参数
        mode = get(params,'mode', 'linear')
        bias = get(params,'bias', 0)
        weight = get(params, 'weight',1)
        power = get(params,'power',1)
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        if mode=='linear':
            output_img = input_img * weight + bias
        elif mode=='exp':
            output_img = input_img ** power
        output_img = np.where(output_img>255,255,output_img)
        # max_val = np.max(output_img)
        # output_img = output_img * 1000 / max_val
        output_img = output_img.astype(np.uint8)
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')

    def _bilateral(self,params):
        d=get(params,'d',0)
        sigma_color=get(params,'sigma_color',100)
        sigma_space=get(params,'sigma_space',15)
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        output_img = cv2.bilateralFilter(input_img, d=d,sigmaColor = sigma_color, sigmaSpace = sigma_space)
        ##############################
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        ##############################
        # TODO 这里也需要修改打印输出时怎么样的
        ##############################
        if test(params, 'verbose',True):
            other = '操作为%s，' % (sys._getframe().f_code.co_name,)
            res = self._base_verbose(params,other)
            dump(res,abbr='hyk')
            
    def _dump_params(self,params):
        dump_simple(self.params[0])

    def _build_graph(self):
        pass

    '''
    更新这个图片集合
    '''
    def update(self):
        # methods = self.methods.copy()
        # dump(methods[0]['readIn'])
        # methods: list
        # method: dict
        self.tree = Tree()
        self.tree.add('hello','')
        for method in self.methods:  
            for key,val in method.items():
                input_label = get(val,'input_label','')
                output_label = get(val,'output_label','')
                if output_label != '' and input_label != '':
                    self.tree.add(output_label,input_label)
                elif output_label =='':
                    # 表示只有输入，也就是这个是writeOut
                    self.tree.color(input_label,'green')
                elif input_label == '':
                    # 只有输出的话，那应该就是第一个节点了
                    self.tree.color(output_label,'red')
                getattr(self,'_'+key)(val)
        print(self.tree.colors)

    def _tree(self,param):
        self.tree.dump()

    '''
    入口函数，开始运行这个函数
    '''
    def run(self, filename = None):
        if filename==None:
            filename = self.filename
        files = CVBase.read_files(filename)
        self.labels = [{}]*len(files)
        self.params = [{}]*len(files)
        for file in files:
            self.file = file
            self.basename = os.path.basename(file)
            if self.basename.split('.')[-1] =='DS_Store':
                continue
            self.update()
            self.index+=1

if __name__ == "__main__":
    hello = CVBase('./pics2','./demo.yaml')
    hello.run()
