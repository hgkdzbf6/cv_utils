import sys
import numpy as np
'''
这个是一些常用的静态方法，先写在这里
'''


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
更加漂亮的输出
'''
def dump(value, *kwargs, abbr='0gk', display = 'normal', color = 'red' ,backcolor = 'black', sep=' ', end='\n', file=sys.stdout, flush=False):
    color_dict = {
        'black':30,
        'red':31,
        'green':32,
        'yellow':33,
        'blue':34,
        'magenta':35,
        'cran':36,
        'white':37
    }
    backcolor_dict = {
        'black':40,
        'red':41,
        'green':42,
        'yellow':43,
        'blue':44,
        'magenta':45,
        'cran':46,
        'white':47
    }
    display_dict = {
        'normal': 0,
        'highlight':1,
        'nobold':22,
        'underline':4,
        'nounderline':24,
        'shine':5,
        'noshine':25,
        'reverse':7,
        'noreverse':27
    }
    abbr_method_dict = {
        '0': 0 ,
        'h': 1,
        'b': 2,
        'B': 22,
        'u': 4,
        'U': 24,
        's': 5,
        'S': 25,
        'r': 7,
        'R': 27,
    }
    abbr_color_dict = {
        'k': 0,
        'r': 1,
        'g': 2,
        'y': 3,
        'b': 4,
        'm': 5,
        'c': 6,
        'w': 7,
    }
    if abbr !='':
        color = abbr_color_dict[abbr[1]] + 30
        backcolor = abbr_color_dict[abbr[2]] + 40
        display = abbr_method_dict[abbr[0]]
        color_cmd = str(color)
        backcolor_cmd = str(backcolor)
        display_cmd = str(display)
    else:
        color_cmd = str(color_dict[color])
        backcolor_cmd = str(backcolor_dict[backcolor])
        display_cmd = str(display_dict[display])
        
    cmd = '\033[' +display_cmd + ';'+ color_cmd +';'+ backcolor_cmd + 'm'
    clean = '\033[0m'
    print(cmd,end='')
    print(value,sep=sep,end='',file=file,flush=flush)
    for arg in kwargs:
        print(arg,sep=sep,end='',file=file,flush=flush)
    print(clean,end=end)

def _is_container(value):
    if isinstance(value,list) or isinstance(value,tuple):
        return 1
    elif isinstance(value,np.ndarray):
        return 2
    elif isinstance(value,dict):
        return 3
    return 0

def dump_list(value):
    if len(value)>10:
        dump('长度为：',len(value),sep='',end='')
    else:
        dump(value,sep='',end='')

def dump_dict(value):
    for key,value in value.items():
        dump_simple(key,': ',value,sep='',end='')
        dump('')
        # dump(key,':',value)

def dump_np(value):
    if max(value.shape) > 10:
        dump('大小为',value.shape,sep='',end='')
    else:
        dump(value,sep='',end='')

def dump_simple(value, *kwargs,sep = '',end=''):
    v = _is_container(value)
    if v==1:
        dump_list(value)
    elif v==2:
        dump_np(value)
    elif v==3:
        dump_dict(value)
    else:
        dump(value,sep=sep,end='')

    for item in kwargs:
        v = _is_container(item)
        if v==1:
            dump_list(item)
        elif v==2:
            dump_np(item)
        elif v==3:
            dump_dict(item)
        else:
            dump(item,sep=sep,end='')

if __name__ == "__main__":
    dump('hello world')