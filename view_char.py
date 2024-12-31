
import os


char_file = 'lib/dataset/txt/char_std_5990.txt'
# , encoding='utf-8'
# , encoding='gbk'
with open(char_file, 'r', encoding='gb18030') as fi:
    label = 0
    # .decode('utf-8')
    s = fi.read()
    print(type(s), len(s))
    words = s.split()
    # print(words)
    print(len(words))
    char_list_map = {w: i for i, w in enumerate(words)}
    print(len(char_list_map))
    # print(char_list_map)
    # words.sort()
    # print(words)
    '''
    for line in fi:
        char_list_map[line.strip()] = label
        label += 1
    '''

print(char_list_map)
print(len(char_list_map))

with open(char_file, 'rb') as file:
    char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
print(len(char_dict))
print(char_dict)

