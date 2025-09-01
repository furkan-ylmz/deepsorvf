# -*- coding:utf8 -*-
import os
from PIL import Image
from shutil import copyfile, copytree, rmtree, move
PATH_DATASET = './car-dataset' 
PATH_NEW_DATASET = './car-reid-dataset' 
PATH_ALL_IMAGES = PATH_NEW_DATASET + '/all_images'
PATH_TRAIN = PATH_NEW_DATASET + '/train'
PATH_TEST = PATH_NEW_DATASET + '/test'
def mymkdir(path):
    path = path.strip() 
    path = path.rstrip("\\") 
    isExists = os.path.exists(path) 
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False
class BatchRename():
    '''
    '''
    def __init__(self):
        self.path = PATH_DATASET 
    def resize(self):
        for aroot, dirs, files in os.walk(self.path):
            filelist = files  
            # print('list', list)
            total_num = len(filelist)  
            for item in filelist:
                if item.endswith('.jpg'):  
                    src = os.path.join(os.path.abspath(aroot), item)
                    im = Image.open(src)
                    out = im.resize((128, 256), Image.ANTIALIAS)  # resize image with high-quality
                    out.save(src)  
    def rename(self):
        for aroot, dirs, files in os.walk(self.path):
            filelist = files  
            # print('list', list)
            total_num = len(filelist)  
            i = 1  
            for item in filelist:
                if item.endswith('.jpg'):  
                    src = os.path.join(os.path.abspath(aroot), item)
                    dirname = str(item.split('_')[0])
                    #new_dir = os.path.join(self.path, '..', 'bbox_all', dirname)
                    new_dir = os.path.join(PATH_ALL_IMAGES, dirname)
                    if not os.path.isdir(new_dir):
                        mymkdir(new_dir)
                    num_pic = len(os.listdir(new_dir))
                    dst = os.path.join(os.path.abspath(new_dir),
                                       dirname + 'C1T0001F' + str(num_pic + 1) + '.jpg')
                    try:
                        copyfile(src, dst) #os.rename(src, dst)
                        print ('converting %s to %s ...' % (src, dst))
                        i = i + 1
                    except:
                        continue
            print ('total %d to rename & converted %d jpgs' % (total_num, i))
    def split(self):
        #---------------------------------------
        #train_test
        images_path = PATH_ALL_IMAGES
        train_save_path = PATH_TRAIN
        test_save_path = PATH_TEST
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
            os.mkdir(test_save_path)
        for _, dirs, _ in os.walk(images_path, topdown=True):
            for i, dir in enumerate(dirs):
                for root, _, files in os.walk(images_path + '/' + dir, topdown=True):
                    for j, file in enumerate(files):
                        if(j==0): 
                            src_path = root + '/' + file
                            dst_dir = test_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
                        else:
                            src_path = root + '/' + file
                            dst_dir = train_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
        rmtree(PATH_ALL_IMAGES)
if __name__ == '__main__':
    demo = BatchRename()
    demo.resize()
    demo.rename()
    demo.split()
