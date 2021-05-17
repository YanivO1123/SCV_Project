import pandas as pd

# c1 = lower left corner
# c2 = upper left corner
# c3 = upper right corner
# c4 = lower right corner

list_ara2012 = ["%03d" % i for i in range(1, 121)]
list_ara2013_canon = ["%03d" % i for i in range(1, 166)]


for i in list_ara2013_canon:
    print(i)

    data = pd.read_csv("Plant_Phenotyping_Datasets/Ara2013-Canon/ara2013_plant" + i + "_bbox.csv")
    data.columns =['index', 'c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y']

    f= open("imgs/Ara2013-Canon/ara2013_plant" + i + "_bbox.txt","w+")

    for i in range(len(data.index)):
        x_coordinate = data.iloc[i]['c2x']
        y_coordinate = data.iloc[i]['c2y']
        width = data.iloc[i]['c3x'] - data.iloc[i]['c2x']
        height = data.iloc[i]['c2y'] - data.iloc[i]['c1y']
        dataForYolo = [x_coordinate, y_coordinate, width, height]
        f.write("%d %d %d %d\n" % (dataForYolo[0], dataForYolo[1], dataForYolo[2], dataForYolo[3]))
    f.close()
