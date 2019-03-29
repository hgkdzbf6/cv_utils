from utils import get, dump

class Forest(object):
    def __init__(self):
        self.dict = {}

class Tree(object):
    def __init__(self):
        self.root = None
        self.dict = {}
        self.colors = {}
    
    def add(self,name,parent_name):
        if not parent_name in self.dict:
            if self.root == None:
                tn = TreeNode(name)
                self.dict[name] = tn
                self.root = tn
            else:
                raise ValueError('这个树已经有根了，不能再插入新的根了qwq')
        else:
            tn = TreeNode(name)
            self.dict[name] = tn
            self.dict[parent_name].children.append(tn)

    def find(self, key):
        pass
    
    def color(self,key,val):
        self.colors[key] = val

    def dump(self):
        if self.root == None:
            return
        else:
            level = 0
            start = self.root           
            color = get(self.colors,self.root.name,'yellow')
            dump(start.name ,sep='',abbr='',color=color)
            self._dump_tree(start,level,[])

    def _dump_tree(self, current, level, is_down):
        right_down = '├'
        straight = '─'
        right = '└'
        down = '│\t'
        if current==None:
            return
        for i,item in enumerate(current.children):
            color = get(self.colors,item.name,'yellow')
            for j in is_down:
                if j == True:
                    dump('\t',end='',abbr='',color=color)
                else:
                    dump(down,end='',abbr='',color=color)
            # 最后一个
            if i == len(current.children) - 1:
                dump(right, straight*7 ,item.name,sep='',abbr='',color=color)
                new_is_down = is_down+[True]
                self._dump_tree(item,level+1,new_is_down)
            else:
                dump(right_down, straight*7 ,item.name,sep='',abbr='',color=color)
                new_is_down = is_down + [False]
                self._dump_tree(item,level+1,new_is_down)
            

class TreeNode(object):
    def __init__(self, name, parent = None):
        self.name = name
        self.parent = parent
        self.children = []

if __name__ == "__main__":
    tree = Tree()
    tree.add('hello','')
    tree.add('world','hello')
    tree.add('hello_world','hello')
    tree.add('nishishui','world')
    tree.add('nishishui2','world')
    tree.add('nishishui3','world')
    tree.add('nishishui4','world')
    tree.add('你是什么鬼？','nishishui4')
    tree.dump()
