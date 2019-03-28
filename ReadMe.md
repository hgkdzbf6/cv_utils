# cv_utils

## 为什么写这个文件？

写这个工具的初衷是使得图像处理能够变得更简单。

## 特点

1. 完全不需要编写代码，只需要更改配置文件当中的数据，处理步骤，输入输出定义好的话，就能够完成图像的批量处理任务。

2. 可扩展性：只要继承`CVBase`这个类，然后重写需要的方法，在方法前面加上下划线，就能够完成相应的操作。

3. 写了一个替代`print`的方法，重命名为`dump`，能够显示更加多彩的画面。

## 文件用法：

### 主要关键字用法：

- input_label:    输入文件的标签
- output_label：  输出文件的标签
- verbose：       显示处理信息。

### readIn

- `mode`: 读入的模式，`0`是黑白，`-1`是原图，emmmm其实就是opencv的啦
- `type`: 读入的格式。因为之前处理过raw格式的文件，所以也一并写进去了。
- `raw_size`: 读入图片的大小
- `save`: 历史遗留问题，删掉没啥的

``` yaml
- readIn:
    mode: 0                   # 图像模式
    type: 'jpg'               # 主要看是不是raw文件
    raw_size: 3072
    output_label: 'hello'     # 保存成一个另外的标签
    verbose: true             # 显示处理结果
```

### print

- `mode`: 目前只有一个图片的`shape`

``` yaml
- print:
    input_label: 'hello'
    mode: 'shape'
    verbose: true
```

### threshold

- `val`: 门限值

### morphology

- `kernel_size` 就是形态学操作的核大小
- `run` 主要操作都放在这里，里面的操作时一个列表
  - `erode_time`: 腐蚀迭代次数
  - `dilate_time`: 膨胀迭代次数
  - `times`: 总循环次数 
  - `res`: 操作名称
    - 0: erode-dilate
    - 1: dilate-erode
    - 2: erode+dilate
    - 3: -erode-dilate

``` yaml
- morphology:
    input_label: 'threshold'
    kernel_size: 5
    run:
        - 
            erode_time: 1
            dilate_time: 1
            times: 1
            res: 0
        - 
            erode_time: 1
            dilate_time: 1
            times: 1
            res: 0
    output_label: 'morphology'
```

### drawColorHist

- `channel`: 通道数
- `threshold`: 在这个值的范围上加一个竖线qwq虽然没啥卵用
- `smooth`: 平滑
- `smooth_param:` 平滑参数

```yaml
- drawColorHist:
    input_label: 'hello'
    channel: 25
    threshold: 50
    smooth: true
    smooth_param: 10
    output_label: 'hist'
    verbose: true
```

### writeOut

- `suffix`: 文件名后缀
- `path`: 文件保存路径

```yaml
- writeOut:
    input_label: 'hist'
    suffix: '_hist'
    path: './imgs'
    verbose: true
```

### drawContour

- `max_length`: 最大长度，表示最大长度
- `min_length`: 最小长度，比这个长度小的话就是杂乱的东西了
- `min_one_size`: 一个元件的最小面积
- `one_size`: 一个元件的一般面积
- `more_size`: 粘连在一起的元件面积
- `circle_size`: 中心圆盘的面积大小
- `mode`: 一个是`height`，一个是`width`。我们以`width`作为宽的哪一部分，`height`作为短的哪一部分。

```yaml
- drawContour:
    input_label:    'hello'
    max_length:     1000
    min_length:     7
    min_one_size:   45
    one_size:       100
    more_size:      1000
    circle_size:    50000
    mode:           'height' # 另一个是'height'
    output_label:   'contour'
    verbose:        true
```

### copy

没啥好说的，就是吧input_label的图片复制到output_label当中，可以作为其他的模板。