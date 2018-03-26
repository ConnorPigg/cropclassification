import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pickle

plt.switch_backend("agg")
gdal.UseExceptions()

simple_labels = None
try:
    print("loading labels")
    simple_labels = pickle.load(open('SimpleLabels','rb'))
except IOError as e:
    print("failed to load labels")
if simple_labels is None:
    labeled_north_raster = gdal.Open("CDL_2013_Champaign_north.tif",\
            gdal.GA_ReadOnly)
    labels = labeled_north_raster.GetRasterBand(1).ReadAsArray()
    print("Labels shape: " + str(labels.shape))
    # only needed once
    #plt.imsave("labeled.png", arr=labels)
    labeled_north_raster = None
    simple_labels = []
    for row in labels:
        for label in row:
            simple_labels.append(label if label == 5 or label == 1 else 0)
    simple_labels = np.array(simple_labels)
    print("Simple labels shape: " + str(simple_labels.shape))
    pickle.dump(simple_labels, open('SimpleLabels','wb'))

################
#### North Raster
################
north_bands_data = None
try:
    print("loading north bands")
    north_bands_data = pickle.load(open('NorthBands','rb'))
except IOError as e:
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
    print("North Bands Shape: " + str(north_bands_data.shape))
    north_bands_data = north_bands_data.reshape((rows * cols, n_bands))
    pickle.dump(north_bands_data, open('NorthBands','wb'))

#print("North shape: " + str(north_bands_data.shape))
#print("band max min")
#print(0, np.max(north_bands_data[:,:,0]), np.min(north_bands_data[:,:,0]))
#print(1, np.max(north_bands_data[:,:,1]), np.min(north_bands_data[:,:,1]))
#print(2, np.max(north_bands_data[:,:,2]), np.min(north_bands_data[:,:,2]))
#print(3, np.max(north_bands_data[:,:,3]), np.min(north_bands_data[:,:,3]))
#print(4, np.max(north_bands_data[:,:,4]), np.min(north_bands_data[:,:,4]))

################
#### South Raster
################
south_bands_data = None
try:
    print("loading south bands")
    south_bands_data = pickle.load(open('SouthBands','rb'))
except IOError as e:
    print("south bands load failed")

#if south_bands_data is None:
#    south_raster = gdal.Open("20130824_RE3_3A_Analytic_Champaign_south.tif",\
#            gdal.GA_ReadOnly)
#    south_bands_data = []
#    for b in range(1, south_raster.RasterCount + 1):
#        band = south_raster.GetRasterBand(b)
#        south_bands_data.append(band.ReadAsArray())
#    south_bands_data = np.dstack(south_bands_data)
#    south_raster = None
#    rows, cols, n_bands = south_bands_data.shape
#    print("South Bands Shape: " + str(south_bands_data.shape))
#    south_bands_data = south_bands_data.reshape((rows * cols, n_bands))
#    pickle.dump(south_bands_data, open('SouthBands','wb'))

#print("South shape: " + str(south_bands_data.shape))
#print("band max min")
#print(0, np.max(south_bands_data[:,:,0]), np.min(south_bands_data[:,:,0]))
#print(1, np.max(south_bands_data[:,:,1]), np.min(south_bands_data[:,:,1]))
#print(2, np.max(south_bands_data[:,:,2]), np.min(south_bands_data[:,:,2]))
#print(3, np.max(south_bands_data[:,:,3]), np.min(south_bands_data[:,:,3]))
#print(4, np.max(south_bands_data[:,:,4]), np.min(south_bands_data[:,:,4]))

#r = south_bands_data[:,:,4]
#g = south_bands_data[:,:,3]
#b = south_bands_data[:,:,2]
#a = south_bands_data[:,:,1]
#u = south_bands_data[:,:,0]
#plt.imsave(fname="south4.png", arr=r)
#plt.imsave(fname="south3.png", arr=g)
#plt.imsave(fname="south2.png", arr=b)
#plt.imsave(fname="south1.png", arr=a)
#plt.imsave(fname="south0.png", arr=u)

#####################
##### Data cleaning
####################
# noticed some blank pixels ([0,0,0,0,0]) which took up about 3% of both the
# top and south images. These pixels were still labeled as corn or soy.
# I'm considering these as dirty/uninformative data and will remove for
# training and testing and assume a blank pixel to be 'other'
#cleaned_north = None
#cleaned_south = None
#cleaned_labels = None
#try:
#    print("loading cleaned data")
#    cleaned_north = pickle.load(open("CleanedNorth","rb"))
##    cleaned_south = pickle.load(open("CleanedSouth","rb"))
#    cleaned_labels = pickle.load(open("CleanedLabels","rb"))
#except IOError as e:
#    print("failed to load a cleaned data set.")
#
#if cleaned_north is None or cleaned_labels is None:
##if cleaned_north is None or cleaned_south is None or cleaned_labels is None:
#    print("Cleaning north data..")
#    cleaned_north = []
#    cleaned_south = []
#    cleaned_labels = []
#    for i in range(len(north_bands_data)):
#        if north_bands_data[i].any():
#            cleaned_north.append(north_bands_data[i])
#            cleaned_labels.append(simple_labels[i])
#    cleaned_north = np.array(cleaned_north)
#    cleaned_labels = np.array(cleaned_labels)
##    print("Cleaning south data..")
##    for i in range(len(south_bands_data)):
##        if south_bands_data[i].any():
##            cleaned_south.append(south_bands_data[i])
##    cleaned_south = np.array(cleaned_south)
#    pickle.dump(cleaned_north, open("CleanedNorth","wb"))
##    pickle.dump(cleaned_south, open("CleanedSouth","wb"))
#    pickle.dump(cleaned_labels, open("CleanedLabels","wb"))

#####################
#### Plotting
####################
# I dont know what to plot yet


##########################################
################ Machine Learning
##########################################
train_x, test_x, train_y, test_y = train_test_split(north_bands_data,\
        simple_labels, shuffle=True, random_state=42)
try:
    print("loading cleaned train data...")
    train_x = pickle.load(open("TrainX","rb"))
    train_y = pickle.load(open("TrainY","rb"))
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
    pickle.dump(train_x, open("TrainX","wb"))
    pickle.dump(train_y, open("TrainY","wb"))

################
#### grid search
################


best_pipe = None
try:
    print("loading best pipe")
    best_pipe = pickle.load(open("BestPipe","rb"))
except IOError as e:
    print("failed to load best pipe")

if best_pipe is None:
# decided on alpha = .001 and hinge method
    #params = {
    #            'sgdclassifier__alpha': 10.0**-np.arange(1,7),
    #            #'sgdclassifier__loss': ['hinge','modified_huber'],
    #        }
################
# the final model
    sgd = SGDClassifier(shuffle=True, loss='hinge', n_jobs=4,\
            alpha=0.001, max_iter=10**3, tol=1e-3)
# to use without searching
    #sgd = SGDClassifier(shuffle=True, loss='hinge', n_jobs=4,\
            #max_iter=10**3, tol=1e-3)
    #sgd = SGDClassifier(shuffle=True, loss='modified_huber',\
            #n_jobs=3, max_iter=10**4, tol=1e-3)
################
# seperate experiment decided on hinged
#    sgd = SGDClassifier(shuffle=True, max_iter=10**3, n_jobs=4, tol=1e-3)
###
    pipe = make_pipeline(StandardScaler(), sgd)
################
# to use with searching
    #search = GridSearchCV(estimator=pipe, param_grid=params, n_jobs=4,\
    #        refit=True, cv=2, scoring='f1_weighted')
    ##search = RandomizedSearchCV(estimator=pipe, param_distributions=params,\
    ##        n_jobs=4, refit=True, cv=2, n_iter=3, scoring='f1_weighted')
    #print("trying to grid search the pipe")
    #search.fit(train_x, train_y)
    #best_pipe = search.best_estimator_
    #print("Search's best params: ")
    #print(search.best_params_)
    #print("Search's best score: ")
    #print(search.best_score_)
###############
# to use without searching
    print("No grid search. Final params decided.")
    best_pipe = pipe
    best_pipe.fit(train_x, train_y)
###############
    pickle.dump(best_pipe, open('BestPipe','wb'))

print("Params")
print(best_pipe.get_params())

random_pred = []
for i in range(len(test_y)):
    random_pred.append(np.random.choice([0,1,5], replace=True))

un, c = np.unique(test_y, return_counts=True)
freq = c/np.sum(c)
weighted_pred = []
for i in range(len(test_y)):
    weighted_pred.append(np.random.choice([0,1,5], replace=True, p=freq))

prediction = best_pipe.predict(north_bands_data)
train_pred = best_pipe.predict(train_x)
test_pred = best_pipe.predict(test_x)
report = classification_report(simple_labels, prediction)
train_report = classification_report(train_y, train_pred)
test_report = classification_report(test_y, test_pred)
random_report = classification_report(test_y, random_pred)
weighted_report = classification_report(test_y, weighted_pred)
print("Report")
print(report)
print("Train Report")
print(train_report)
print("Test Report")
print(test_report)
print("Random Report")
print(random_report)
print("Weighted Random Report")
print(weighted_report)



################
#### non-grid
################

#clf = None
#try:
#    print("loading clf")
#    clf = pickle.load(open('CLF','rb'))
#except IOError as e:
#    print("clf failed to load.")
#    print(str(e))
#if clf is None:
#    print("making classifier")
#    clf = SGDClassifier(shuffle=True,\
#            n_iter=np.ceil(10**6 / len(train_labels))\
#            )
#    clf.fit(train_features, train_lables)
#    # save the model
#    pickle.dump(clf, open('CLF','wb'))

# the now fitted tree can be scored and used on the south part of the image
#print("testing")
#test_prediction = clf.predict(test_features)
#print("generating reports")
#report = metrics.classification_report(test_labels, test_prediction)
#print(report)
#print("Parameters of CLF:")
#print(clf.get_params())
print("Got to end")
