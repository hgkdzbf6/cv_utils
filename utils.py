import os
import os.path as ops
import cv2
import numpy as np

def init_file(img_path):
    files = os.listdir(img_path)
    files_arr=[]
    for file_name in files:
        full_name = ops.join(img_path,file_name)
        if ops.isfile(full_name):
            if full_name.split('.')[-1] == 'jpg' or full_name.split('.')[-1] == 'JPG':
                files_arr.append(full_name)
    files_arr.sort()
    return files_arr

class CV_Utils(object):
    def __init__(self,img_path, output_path):
        self.channel=25
        self.hists={}
        self.file_name = img_path.split('/')[-1]
        self.output_path = output_path
        if not ops.exists(output_path):
            os.makedirs(output_path)
        if ops.isdir(img_path):
            self.files=[]
            self.init_file(img_path)
            for i in range(len(self.files)):
                self.img = cv2.imread(ops.join(img_path,self.files[i]),0)
                self.file_name = self.files[i]
                self.main(show=False)
        if ops.isfile(img_path):
            self.img = cv2.imread(img_path,0)
            self.main(show=False)

    def init_file(self, img_path):
        files = os.listdir(img_path)
        for file_name in files:
            full_name = ops.join(img_path,file_name)
            if ops.isfile(full_name):
                if full_name.split('.')[-1] == 'jpg' or full_name.split('.')[-1] == 'JPG':
                    self.files.append(full_name)
        self.file_index= -1
        self.files.sort()

    def draw_color_hist(self,the_name,image, color,draw=True, threshold=-1, smooth=False,smooth_param=10): 
        histImg = np.zeros([256,256,3], np.uint8) 
        hist_list=[]
        if smooth==False:
            hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])    
            _ , maxVal, _ , _ = cv2.minMaxLoc(hist)      
            hpt = int(0.9* 256)    
            for h in range(256):    
                intensity = int(hist[h]*hpt/maxVal) 
                hist_list.append(intensity)
                cv2.line(histImg,(h,256), (h,256-intensity), color)    
            if threshold > 0 :
                cv2.line(histImg,(threshold,256), (threshold,0), [255,255,0]) 
            self.hists[the_name]=hist_list
        
        if smooth == True:
            temp=self.hists[the_name]
            # 算数平均
            for i in range(smooth_param,len(temp)-smooth_param):
                sum=0
                for j in range(-smooth_param,smooth_param+1):
                    sum=sum+self.hists[the_name][i+j]
                temp[i]=sum/2.0/smooth_param
            for h in range(256):    
                cv2.line(histImg,(h,256), (h,int(256-temp[h])), color)   
            return histImg, temp

        return histImg ,hist_list

    def draw_hist_diff(self, hist_list, factor=10):
        temp=hist_list
        # 算数平均
        for i in range(factor,len(hist_list)-factor):
            sum=0
            for j in range(-factor,factor+1):
                sum=sum+hist_list[i+j]
            temp[i]=sum/2.0/factor
        hist_list=temp
        histImg = np.zeros([256,256,3], np.uint8)    
        hist_diff = []
        for h in range(255):
            hist_diff.append(temp[h+1]-temp[h])
        the_max=max(hist_diff)
        hist_diff=np.array(hist_diff)
        hist_diff = hist_diff/the_max*128
        # +-128范围内
        for i in range(254):
            if abs(hist_diff[i]-hist_diff[i+1])>30:
                hist_diff[i+1] = hist_diff[i]
        for h in range(255):    
            cv2.line(histImg,(h,256), (h,int(128 + hist_diff[h] )), [255,255,0])   

        cv2.line(histImg,(0,128), (256,128), [255,0,255])    
        return histImg, hist_diff

    def draw_hist_ddiff(self, hist_diff, factor=10):
        temp=hist_diff
        # 算数平均
        for i in range(factor,len(hist_diff)-factor):
            sum=0
            for j in range(-factor,factor+1):
                sum=sum+hist_diff[i+j]
            temp[i]=sum/2.0/factor
        hist_diff=temp
        histImg = np.zeros([256,256,3], np.uint8)    
        hist_ddiff = []
        for h in range(254):
            hist_ddiff.append(temp[h+1]-temp[h])
        the_max=max(hist_ddiff)
        hist_ddiff=np.array(hist_ddiff)
        hist_ddiff = hist_ddiff/the_max*128
        # +-128范围内
        for i in range(253):
            if abs(hist_ddiff[i]-hist_ddiff[i+1])>32:
                hist_ddiff[i+1] = hist_ddiff[i]
        for h in range(254):    
            cv2.line(histImg,(h,256), (h,int(128 + hist_ddiff[h] )), [255,255,0])   

        cv2.line(histImg,(0,128), (256,128), [255,0,255])    
        return histImg,hist_ddiff
        

    def find_best_point(self, hist_list, hist_diff, hist_ddiff):
        # 一阶导数等于0,二阶导数大于0,找两个点
        _ , local_min = self.find_peak(hist_diff, hist_ddiff)
        if len(local_min)>0:
            return local_min[int(len(local_min)/2)]
        return 0    
    
    def find_best_point2(self, maximum_value, average_value):
        return ( - maximum_value + average_value)* 3 + average_value

    def find_peak(self, hist_diff, hist_ddiff):        
        # 一阶导数等于0,二阶导数大于0,找两个点
        local_max=[]
        local_min=[]
        for i in range(253):
            if hist_diff[i] * hist_diff[i+1] < 0:
                if hist_ddiff[i] < 0 and hist_ddiff[i+1] < 0 :
                    local_max.append(i)
                if hist_ddiff[i] > 0 and hist_ddiff[i+1] > 0 :
                    local_min.append(i)
        return local_max,local_min

    def average_lightness(self, img):
        return img.mean()

    def hist_max(self,hist_list, hist_diff, hist_ddiff):
        local_max , _ = self.find_peak(hist_diff, hist_ddiff)
        global_max = 0
        global_max_index = -1
        for i in range(len(local_max)):
            if global_max < hist_list[local_max[i]]:
                global_max = hist_list[local_max[i]]
                global_max_index = local_max[i]
        return global_max_index

    def main(self, show= True):
        # b, g, r = cv2.split(self.img)
        # histImgB = self.draw_color_hist("histImgB",b, [255, 0, 0])    
        # histImgG = self.draw_color_hist("histImgG",g, [0, 255, 0])    
        # histImgR = self.draw_color_hist("histImgR",r, [0, 0, 255])   
        histImg, _ = self.draw_color_hist("histImg",self.img,[255,255,255]) 
        histImg, hist_list = self.draw_color_hist("histImg",self.img,
            [255,255,255],threshold=0,smooth=True, smooth_param=4)

        histDiff, hist_diff=self.draw_hist_diff(hist_list)
        histDDiff,hist_ddiff=self.draw_hist_ddiff(hist_diff)
        threshold = self.find_best_point(hist_list,hist_diff,hist_ddiff)
        threshold = 103
        average= self.average_lightness(self.img)
        # print(average)
        hist_max = self.hist_max(hist_list,hist_diff,hist_ddiff)
        # print(hist_max)
        threshold = self.find_best_point2(hist_max, average)
        # print(threshold)
        gray_img = self.img
        _,bin_img= cv2.threshold(gray_img,threshold,255,cv2.THRESH_BINARY)
        # cv2.imshow("histImgB", histImgB)    
        # cv2.imshow("histImgG", histImgG)    
        # cv2.imshow("histImgR", histImgR)
        if show:
            cv2.namedWindow("histImg",cv2.WINDOW_NORMAL)
            cv2.namedWindow("histDiff",cv2.WINDOW_NORMAL)
            cv2.namedWindow("histDDiff",cv2.WINDOW_NORMAL)
            cv2.namedWindow("Img",cv2.WINDOW_NORMAL)
            cv2.namedWindow("BinImg",cv2.WINDOW_NORMAL)
            cv2.imshow("histImg", histImg)    
            cv2.imshow("histDiff", histDiff) 
            cv2.imshow("histDDiff", histDDiff)    
            cv2.imshow("Img", self.img)    
            cv2.imshow("BinImg", bin_img)    
            cv2.waitKey(0)    
            cv2.destroyAllWindows()   
        else:
            print(self.file_name)
            cv2.imwrite(ops.join(self.output_path,self.file_name),bin_img)

if __name__ == '__main__':
    utils=CV_Utils('./','./output')