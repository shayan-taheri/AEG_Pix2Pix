import glob

num_list_str = []
num_list = []

for name in glob.glob('/home/shayan/Adv_Dataset/Test/CW/Unfooling_Data/*'):
    num_list_str.append(name[54:len(name)-16])
    num_list.append(int(name[54:len(name)-16]))

num_list.sort()

for ix in range(0,len(num_list)):
    print(ix)
    print(num_list[ix])
    if (ix != num_list[ix]):
        print('ERROR')
