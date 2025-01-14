import os, shutil
lis = os.listdir('sample/image')
length = len(lis)
print(lis)
for i in range(0, length):
    shutil.copyfile('sample/image/{}'.format(lis[i]), 'sample/_sample/dateset{}/{}'.format(i, lis[i]))
# os.makedirs('sample/_sample/dateset0')