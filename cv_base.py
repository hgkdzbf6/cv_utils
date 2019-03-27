
import os
import os.path as ops
import cv2
import numpy as np
import yaml
import sys

'''
得到params当中的参数
'''
def get(params, val, default = ''):
    if val in params:
        return params[val]
    else:
        return default
'''
测试val的值是不是test_val
'''
def test(params, val, test_val):
    if val in params:
        return params[val] == test_val
    else:
        return False

'''
测试val的值是不是test_val
'''
def exist(params, val):
    if val in params:
        return True
    else:
        return False

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
        self.index = 0
        self.methods = []
        self.labels = []
        self.hists = []
        self.filename = filename
        self.config = config
        self.parse(self.config)
        pass

    '''
    从参数文件当中读取一些数据，放在methods里面
    '''
    def parse(self, filename):
        f = open(filename, 'r')
        self.methods = yaml.load(f, Loader=yaml.FullLoader)
        # print(self.methods)

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
        # print(dirname)
        for file_name in files:
            full_name = ops.join(dirname,file_name)
            if ops.isfile(full_name):
                if filter == None or full_name.split('.')[-1] in filter:
                    files_arr.append(full_name)
        files_arr.sort()
        return files_arr

    '''
    把路径变成图片
    '''
    @staticmethod
    def readIn(file, mode = -1):
        img = cv2.imread(file,mode)
        return img

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
        pic = CVBase.readIn(self.file, mode=mode)
        if get(params,'save',True):
            label = get(params, 'output_label', 'img'+str(self.index))
            self.labels[self.index][label] = pic
            if test(params, 'verbose', True):
                other = '叫做%s的图片读入了，'%(self.basename,)
                res = self._base_verbose(params,other)
                print(res)

    def _print(self, params):
        if test(params, 'mode', 'shape'):
            if exist(params, 'input_label'):
                label_name = get(params, 'input_label', 'hello')
                test_pic = self.labels[self.index][label_name]
                if test(params, 'verbose', True):
                    print('第%d张图片，标签为%s的这张图片的高是%s，宽是%s。'%(self.index, label_name, test_pic.shape[0],test_pic.shape[1]))
                else:
                    print(self.labels[self.index][label_name].shape)

    def _threshold(self, params):
        if exist(params, 'input_label'):
            input_name = get(params, 'input_label', 'hello')
        origin_pic = self.labels[self.index][input_name]
        val = get(params,'val',1)
        ret,pic = cv2.threshold(origin_pic,val,255, cv2.THRESH_BINARY)
        if params['save']==True:
            if exist(params, 'output_label'):
                label_name = get(params, 'output_label', 'hello')
                self.labels[self.index][label_name] = pic
        if test(params, 'verbose', True):
            other = '门限参数为%d，' % (val,)
            res = self._base_verbose(params,other)
            print(res)
    
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
            other = '操作为%s，保存路径为%s，' % (sys._getframe().f_code.co_name,out_name,)
            res = self._base_verbose(params,other)
            print(res)

    def _writeFile(self, params):
        pass

    '''
    画颜色直方图，这个一般是黑白图，注意二值图不行，一般是拿这个结果，找到二值图的阈值的。
    '''
    def _drawColorHist(self, params):
        histImg = np.zeros([256,256,3], np.uint8)
        hist_list = []
        color = (255,255,255)
        smooth = get(params, 'smooth')
        threshold = get(params, 'threshold')
        smooth_param = get(params, 'smooth_param',10)
        if exist(params, 'input_label'):
            label_name = get(params, 'input_label', 'hello')
            test_pic = self.labels[self.index][label_name]
        hist = cv2.calcHist([test_pic], [0], None, [256], [0.0,255.0])
        _ , maxVal, _ , _ = cv2.minMaxLoc(hist)  
        hpt = int(0.9* 256)  
        for h in range(256):    
            intensity = int(hist[h]*hpt/maxVal) 
            hist_list.append(intensity)
            cv2.line(histImg,(h,256), (h,256-intensity), color)    
        if threshold > 0 :
            cv2.line(histImg,(threshold,256), (threshold,0), [255,255,0]) 
        self.hists[self.index][label_name] = hist_list

        if smooth == True:
            temp=self.hists[self.index][label_name]
            for i in range(smooth_param,len(temp)-smooth_param):
                sum=0
                for j in range(-smooth_param,smooth_param+1):
                    sum=sum+self.hists[self.index][label_name][i+j]
                temp[i]=sum/2.0/smooth_param
            for h in range(256):    
                cv2.line(histImg,(h,256), (h,int(256-temp[h])), color)   
        if exist(params, 'output_label'):
            output_name = get(params, 'output_label', 'hello')
            self.labels[self.index][output_name] = histImg
        if test(params, 'verbose',True):
            other = '操作为%s，光滑开关为%d，门限为%d，光滑参数为%d，' % (sys._getframe().f_code.co_name,smooth,threshold, smooth_param)
            res = self._base_verbose(params,other)
            print(res)

    '''
    更新这个图片集合
    '''
    def update(self):
        methods = self.methods.copy()
        self.labels.append({})
        self.hists.append({})
        # print(methods[0]['readIn'])
        # methods: list
        # method: dict
        for method in methods:  
            for key,val in method.items():
                getattr(self,'_'+key)(val)

    '''
    入口函数，开始运行这个函数
    '''
    def run(self, filename = None):
        if filename==None:
            filename = self.filename
        files = CVBase.read_files(filename)
        for file in files:
            self.file = file
            self.basename = os.path.basename(file)
            self.update()
            self.index+=1

    
if __name__ == "__main__":
    hello = CVBase('./pics','./config.yaml')
    hello.run()