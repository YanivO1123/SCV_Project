import matplotlib.pyplot as plt

first_box = []

with open('leaf_data/labels/ara2012_plant001_rgb.txt') as f_stat:
    l_stat = f_stat.readlines()
    first_box = l_stat[-1]

xs = [first_box[1], first_box[3]]
ys = [first_box[2], first_box[4]]

plt.figure()
plt.scatter(xs, ys)
plt.show()