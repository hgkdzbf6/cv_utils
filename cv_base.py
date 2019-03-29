
import os
import os.path as ops
import cv2
import numpy as np
import yaml
import rawpy
from PIL import Image
import sys
import matplotlib.pyplot as plt
import math
from utils import *
from tree import Tree


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
        suffix = get(params,'suffix')
        t = filename.split('.')
        t[-1] = suffix + '.' + t[-1]
        filename = ''.join(t)
        out_name = ops.join(path, filename)
        cv2.imwrite(out_name,self.labels[self.index][label_name])
        if test(params, 'verbose', True):
            other = '操作为%s，保存文件名为%s，' % (sys._getframe().f_code.co_name,filename,)
            res = self._base_verbose(params,other)
            dump(res)

    def _writeFile(self, params):
        pass

    def _morphology(self,params):
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
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
                erode = cv2.dilate(temp, kernel, iterations = erode_time)
                dilate = cv2.erode(temp, kernel, iterations = dilate_time)
                if res == 0:
                    temp = erode - dilate
                elif res == 1:
                    temp = dilate - erode
                elif res == 2:
                    temp = erode + dilate
                elif res == 3:
                    temp = - erode - dilate
        output_img = temp
        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        if test(params, 'verbose',True):
            other = '操作为%s，形态学核大小为%d，' % (sys._getframe().f_code.co_name,kernel_size)
            res = self._base_verbose(params,other)
            dump(res)


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
        param_out = get(params, 'param_out','hist')
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
        if threshold > 0 :
            cv2.line(histImg,(threshold,256), (threshold,0), [255,255,0]) 
        self.params[self.index][param_out] = hist_list

        if smooth == True:
            temp=self.params[self.index][param_out]
            for i in range(smooth_param,len(temp)-smooth_param):
                sum=0
                for j in range(-smooth_param,smooth_param+1):
                    sum=sum+self.params[self.index][param_out][i+j]
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

    def _drawContour(self, params):
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]

        # 读入参数
        max_length = get(params,'max_length', 1000) 
        min_length = get(params, 'min_length',7)
        min_one_size = get(params,'min_one_size',45)
        one_size = get(params,'one_size',100)
        more_size = get(params,'more_size',1000)
        circle_size = get(params,'circle_size',50000)
        param_out = get(params,'param_out',{})
        # 找轮廓
        contours, hierarchy = cv2.findContours(input_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        the_res = input_img.copy()
        the_res = cv2.cvtColor(the_res,cv2.COLOR_GRAY2BGR)
        # 找到中心点
        center = list(the_res.shape)
        center[0]/=2
        center[1]/=2
        red_contours = []

        def cross(m1,m2):
            return float(m1[0]*m2[1] - m1[1]*m2[0])

        def vec(p1,p2):
            return [float(p1[0])/100.-float(p2[0])/100., float(p1[1])/100.-float(p2[1])/100.]

        def inBox(box, point):
            A = list(box[0])
            B = list(box[1])
            C = list(box[2])
            D = list(box[3])
            P = point
            AP = vec(A,P)
            AB = vec(A,B)
            CD = vec(C,D)
            CA = vec(C,A)
            ABP = cross(AB,AP)
            CDA = cross(CD,CA)
            one = ABP*CDA
            two = cross(vec(B,C),vec(B,P))*cross(vec(D,A),vec(D,P))
            return (one>0 and two>0)
        
        # 第一次循环，找到圆心
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            # 检验宽度和高度，如果小于10或者大于1000的话跳过
            if rect[1][0] >max_length or rect[1][1]> max_length or rect[1][0] < min_length or rect[1][1] <min_length:
                continue
            # 检验面积
            area=rect[1][0]*rect[1][1]
            box = cv2.boxPoints(rect)
            if area > circle_size and inBox(box, center):
                center_contour = contour
                break
        
        # 找到中心的x和y坐标
        center_rect = cv2.fitEllipse(center_contour)
        # (x,y), (w,h), angle
        self.params[self.index][param_out['center']] = center_rect

        # 第二次循环，找到red
        rects = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            # 检验宽度和高度，如果小于10或者大于1000的话跳过
            if rect[1][0] >max_length or rect[1][1]> max_length or rect[1][0] < min_length or rect[1][1] <min_length:
                continue
            # 检验面积
            area=rect[1][0]*rect[1][1]
            if area < min_one_size:             # 细小的轮廓跳过
                continue
            # 和中心圆盘一样的话跳过
            # if contour.all() == center_contour.all():
            #     continue
            # elif area < more_size:     # 有可能是1个的
            #     green_contours.append(contour)
            #     continue

            # 宽度比高度大
            if rect[1][1]>rect[1][0]:
                w,h = rect[1][1],rect[1][0]
            else:
                w,h = rect[1][0],rect[1][1]
            rects.append([w,h])
            # 其他就是红色的
            box = cv2.boxPoints(rect)
            red_contours.append(contour)
        
        rects = np.array(rects)
        print(rects.shape)
        rects = rects[np.lexsort(rects[:,::-1].T)]
        mean_length = np.mean(rects[:10,:],axis = 0)
        print(mean_length)
        # 找到距离
        font=cv2.FONT_HERSHEY_SIMPLEX
        count = 0
        # 开始画
        for contour in red_contours:
            min_dist = 80000
            max_dist = 0
            for point in contour:
                point = point.tolist()[0]
                # dump(point[0],center_rect[0])
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
            if predict_num == 1:
                color = (0.0,255.0,0.0)
            else:
                color = (0.0,0.0,255.0)
            linewidth = 1
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            for i in range(4):
                cv2.line(the_res,tuple(box[i]), tuple(box[(i+1)%4]), color, linewidth) # 画线
            cv2.putText(the_res,str(predict_num),(int(rect[0][0]),int(rect[0][1])), font, 1,color,1)
            count+=predict_num
        output_img = the_res.copy()
        # 得到了最近距离和最远距离
        self.params[self.index][param_out['count']] = count

        if exist(params, 'output_label'):
            output_label = get(params, 'output_label', 'hello')
            self.labels[self.index][output_label] = output_img
        if test(params, 'verbose',True):
            other = '操作为%s，统计个数为%d，' % (sys._getframe().f_code.co_name,count)
            res = self._base_verbose(params,other)
            dump(res,'hyk')

    '''
    把笛卡尔坐标系变成极坐标系
    '''
    def _toPolar(self,params):
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
            dump(res,'hyk')

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
            dump(res,'hyk')


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
            dump(res,'hyk')

    '''
    使用滤镜
    '''
    def _filter(self,params):
        kernel_type = get(params, 'kernel_type', 'gaussian')
        kernel_size = get(params, 'kernel_size', 3)
        ##############################
        # TODO 这里填写输入参数
        ##############################
        if exist(params, 'input_label'):
            input_label = get(params, 'input_label', 'hello')
            input_img = self.labels[self.index][input_label]
        ##############################
        # TODO 这里填写具体操作
        # output_img = cv2.filter2D()
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
            dump(res,'hyk')

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
            dump(res,'hyk')


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
            dump(res,'hyk')
            
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
    hello = CVBase('./pics2','./config.yaml')
    hello.run()