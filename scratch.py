from cv_base import *

if __name__ == "__main__":
    hello = CVBase('./scratch','./scratch.yaml')
    # hello = CVBase('./speckle','./scratch.yaml')
    hello.run()
