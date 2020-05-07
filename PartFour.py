import os
#pip3 install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/   --trusted-host mirrors.aliyun.com

listdir=os.listdir('d:\\')
print(listdir.__sizeof__())
for f in listdir:
    abdir='d:\\'+f+'\\'
    if os.path.isdir(abdir):
        print("{}是一个目录".format(abdir))
        try:
            next=os.listdir(abdir)
            for iter in next:
                abs=abdir+iter
                if os.path.isfile(abs):
                    dir,file=os.path.split(abs)
                    print('目录名：{}，文件名：{}'.format(dir,file))
        except PermissionError as err:
            print(err)
    else:
        dir,file=os.path.split(abdir)
        print('目录名：{}，文件名：{}'.format(dir,file))
print("end")