import pandas as pd

# c1 = lower left corner
# c2 = upper left corner
# c3 = upper right corner
# c4 = lower right corner

data = pd.read_csv("Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2012/ara2012_plant001_bbox.csv")

data.columns =['index', 'c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y']

f= open("testPrepareData.txt","w+")

for i in range(len(data.index)):
    x_coordinate = data.iloc[i]['c2x']
    y_coordinate = data.iloc[i]['c2y']
    width = data.iloc[i]['c3x'] - data.iloc[i]['c2x']
    height = data.iloc[i]['c2y'] - data.iloc[i]['c1y']
    dataForYolo = [x_coordinate, y_coordinate, width, height]
    f.write("%d %d %d %d\n" % (dataForYolo[0], dataForYolo[1], dataForYolo[2], dataForYolo[3]))

