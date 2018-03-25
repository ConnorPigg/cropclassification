import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import load, dump
import json

gdal.UseExceptions()
plt.switch_backend("agg")

###################
## GLOBAL CONSTANTS
###################
# shape of .tif files to used after pickling
rows, cols = 5959, 9425

##################################
#### Reading Labeled north data
##################################
simple_labels = None
try:
    print("loading labels")
    simple_labels = load(open('SimpleLabels.pickle','rb'))
except:
    print("failed to load labels")

if simple_labels is None:
    labeled_north_raster = gdal.Open("CDL_2013_Champaign_north.tif",\
            gdal.GA_ReadOnly)
    labels = labeled_north_raster.GetRasterBand(1).ReadAsArray()
    labeled_north_raster = None

# Simplify classes to three 0=other 1=corn 5=soybeans
    simple_labels = []
    for row in labels:
        for label in row:
            simple_labels.append(label if label == 5 or label == 1 else 0)
    simple_labels = np.array(simple_labels)
    dump(simple_labels, open('SimpleLabels.pickle','wb'))

################
#### North Raster
################
north_bands_data = None
try:
    print("loading north bands")
    north_bands_data = load(open('NorthBands.pickle','rb'))
except:
    print("north bands load failed")

if north_bands_data is None:
    north_raster = gdal.Open("20130824_RE3_3A_Analytic_Champaign_north.tif",\
            gdal.GA_ReadOnly)
    north_bands_data = []
    for b in range(1, north_raster.RasterCount + 1):
        north_bands_data.append(north_raster.GetRasterBand(b).ReadAsArray())
    north_bands_data = np.dstack(north_bands_data)
    north_raster = None
    rows, cols, n_bands = north_bands_data.shape
    print("north, south, and label shape: ", rows, cols)
    north_bands_data = north_bands_data.reshape((rows * cols, n_bands))
    dump(north_bands_data, open("NorthBands.pickle","wb"))


################
#### South Raster
################
south_bands_data = None
try:
    print("loading south bands")
    south_bands_data = load(open('SouthBands.pickle','rb'))
except:
    print("south bands load failed")

if south_bands_data is None:
    south_raster = gdal.Open("20130824_RE3_3A_Analytic_Champaign_south.tif",\
            gdal.GA_ReadOnly)
    south_bands_data = []
    for b in range(1, south_raster.RasterCount + 1):
        band = south_raster.GetRasterBand(b)
        south_bands_data.append(band.ReadAsArray())
    south_bands_data = np.dstack(south_bands_data)
    south_raster = None
    rows, cols, n_bands = south_bands_data.shape
    south_bands_data = south_bands_data.reshape((rows * cols, n_bands))
    dump(south_bands_data, open('SouthBands.pickle','wb'))

##################################
######## Machine Learning
##################################
train_x, test_x, train_y, test_y = train_test_split(north_bands_data,\
        simple_labels, shuffle=True, random_state=42)
try:
    print("loading cleaned train data...")
    train_x = load(open("TrainX.pickle","rb"))
    train_y = load(open("TrainY.pickle","rb"))
except:
    print("loading cleaned train data failed.")
    print("Cleaning training data..")
    cleaned_y = []
    cleaned_x = []
    for i in range(len(train_x)):
        if train_x[i].any():
            cleaned_x.append(train_x[i])
            cleaned_y.append(train_y[i])
    cleaned_x = np.array(cleaned_x)
    cleaned_y = np.array(cleaned_y)
    train_x = cleaned_x
    train_y = cleaned_y
    dump(train_x, open("TrainX.pickle","wb"))
    dump(train_y, open("TrainY.pickle","wb"))

###########
## Training
###########
pipe = None
try:
    print("loading pipe")
    pipe = load(open("Pipe.pickle","rb"))
except:
    print("failed to load pipe")

if pipe is None:
# hyperparameters previously chosen using GridSearchCV with 2 k-folds
    sgd = SGDClassifier(shuffle=True, loss='hinge', n_jobs=4,\
            alpha=.01, max_iter=10**3, tol=1e-3)
    pipe = make_pipeline(StandardScaler(), sgd)
    pipe.fit(train_x, train_y)
    dump(pipe, open("Pipe.pickle","wb"))

############
### Testing
############

test_pred = pipe.predict(test_x)
report = classification_report(test_y, test_pred)
print("Test set report")
print(report)
score = accuracy_score(test_y, test_pred)
print("Test set accuracy")
print(score)

random_pred = []
for i in range(len(test_x)):
    random_pred.append(np.random.choice([0,1,5]))

u,c = np.unique(random_pred, return_counts=True)
random_f = c/np.sum(c)
u, c = np.unique(test_pred, return_counts=True)
test_pred_f = c/np.sum(c)
u, c = np.unique(test_y, return_counts=True)
test_y_f = c/np.sum(c)

N = 3
size = 0.3
pos = np.arange(N)
fig, ax = plt.subplots()
bar1 = ax.bar(pos, random_f, size, color='red')
bar2 = ax.bar(pos + size, test_y_f, size, color='black')
bar3 = ax.bar(pos + 2*size, test_pred_f, size, color='blue')
ax.set_ylabel("Frequencies")
ax.set_title("Comparing crop distributions")
ax.set_xticks(pos + 3*size/2)
ax.set_xticklabels(("Other", "Corn", "Soybeans"))
ax.legend((bar1[0], bar2[0], bar3[0]),\
        ("Random Choice", "Actual", "Model Predicted"))
plt.savefig("compared_distributions.png")

##################
## Predicting South
##################

south_pred = pipe.predict(south_bands_data)
json.dump(south_pred.tolist(), open("south_pred.json","w"))

#####################
## Converting to tif
#####################

#reshape prediction into col by row instead of a list of pixels
south = south_pred.reshape((rows,cols))
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create("south_pred.tif", cols, rows, 1, gdal.GDT_Byte)
band = dataset.GetRasterBand(1)
band.WriteArray(south)
dataset = None
