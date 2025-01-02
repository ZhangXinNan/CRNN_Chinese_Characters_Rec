
import os
import shutil


in_dir = r'C:\data_public\OCR\CRNN_Chinese_Characters_Rec\images'
out_dir = r'C:\data_public\OCR\CRNN_Chinese_Characters_Rec\images_test'
test_file = 'lib/dataset/txt/test.txt'

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

with open(test_file, 'r') as fi:
    i = 0
    for line in fi:
        arr = line.split(' ')
        img_path = os.path.join(in_dir, arr[0])
        shutil.copy(img_path, out_dir)
        print(i, img_path)
        i += 1
