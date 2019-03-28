class Forest(object):
    def __init__(self):
        self.dict = {}

class Tree(object):
    def __init__(self):
        self.root = None
        self.dict = {}
    
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

    def dump(self):
        if self.root == None:
            return
        else:
            level = 0
            start = self.root
            print(start.name ,sep='')
            Tree._dump_tree(start,level,[])

    @staticmethod
    def _dump_tree(current, level, is_down):
        right_down = '├'
        straight = '─'
        right = '└'
        down = '│\t'
        if current==None:
            return
        for i,item in enumerate(current.children):
            for j in is_down:
                if j == True:
                    print('\t',end='')
                else:
                    print(down,end='')
            # 最后一个
            if i == len(current.children) - 1:
                print(right, straight*7 ,item.name,sep='')
                new_is_down = is_down+[True]
                Tree._dump_tree(item,level+1,new_is_down)
            else:
                print(right_down, straight*7 ,item.name,sep='')
                new_is_down = is_down + [False]
                Tree._dump_tree(item,level+1,new_is_down)
            

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
