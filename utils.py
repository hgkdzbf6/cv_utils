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
def dump(value, display = 'normal', color = 'red' ,backcolor = 'black', reverse = True):
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
        'blink':5,
        'noblink':25,
        'reverse':7,
        'noreverse':27
    }
    color_cmd = str(color_dict[color])
    backcolor_cmd = str(backcolor_dict[backcolor])
    display_cmd = str(display_dict[display])
    cmd = '\033[' +display_cmd + ';'+ color_cmd +';'+ backcolor_cmd + 'm'
    clean = '\033[0m'
    print(cmd,value,clean,sep='')
    

if __name__ == "__main__":
    dump('hello world')