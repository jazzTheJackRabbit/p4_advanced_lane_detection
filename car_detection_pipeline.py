import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.image as mpimg

# Create histogram features
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(image[:,:,0], bins=32, range=(0,255))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0,255))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0,255))
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
   
# Define a function that takes an image, a color space, 
# and a new image size
# and returns a feature vector
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    COLOR = eval("cv2.COLOR_BGR2{}".format(color_space))
    img = cv2.cvtColor(img, COLOR)
    
    # Use cv2.resize().ravel() to create the feature vector
    resized_img = cv2.resize(img, size)
    feature_vector = resized_img.ravel()
    
    # Return the feature vector
    return feature_vector

# Define a function to return HOG features and visualization
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):    
    features, hog_image = hog(
        img, 
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        block_norm='L2',
        visualise=True,
        feature_vector=True
    )
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        return({
            'features':features, 
            'image': hog_image
        })
    else:      
        # Use skimage.hog() to get features only
        return({
            'features':features
        })
    
orient = 9
pix_per_cell = 8
cell_per_block = 2
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

###### TODO ###########
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        image = cv2.imread(img)
        
        # apply color conversion if other than 'RGB'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply bin_spatial() to get spatial color features
        spatial_bin_features = bin_spatial(image)
        
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(image)
        
        hog_features = get_hog_features(gray)['features']    
        # Append the new feature vector to the features list
        feature_vector = np.concatenate((spatially_binned, hist_features, hog_features))
        
        features.append(feature_vector)
        
    # Return list of feature vectors
    return features

def preprocess_data():
	images = glob.glob('../data/vehicle_detection/*.jpeg')
	cars = []
	notcars = []
	for image in images:
	    if 'image' in image or 'extra' in image:
	        notcars.append(image)
	    else:
	        cars.append(image)
	        
	car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
	                        hist_bins=32, hist_range=(0, 256))
	
	notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
	                        hist_bins=32, hist_range=(0, 256))

	if len(car_features) > 0:
	    # Create an array

	    # y stack of feature vectors
	    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

	    # Fit a per-column scaler
	    X_scaler = StandardScaler().fit(X)
	    # Apply the scaler to X
	    scaled_X = X_scaler.transform(X)
	    car_ind = np.random.randint(0, len(cars))

	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	rand_state = np.random.randint(0, 100)

	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, 
	    y, 
	    test_size=0.2, 
	    random_state=rand_state
	)

def train(X_train, y_train, X_test, y_test):
	from sklearn.svm import LinearSVC
	# Use a linear SVC (support vector classifier)
	svc = LinearSVC()
	# Train the SVC
	svc.fit(X_train, y_train)
	print('Test Accuracy of SVC = ', svc.score(X_test, y_test))


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    

def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None), 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    
    x_stop = x_start_stop[1]
    x_search_subset = (x_start_stop[1] - x_start_stop[0])
    x_start = x_stop  - (int(x_search_subset/xy_window[0]) * xy_window[0])
    
    y_stop = y_start_stop[1]
    y_search_subset = (y_start_stop[1] - y_start_stop[0])    
    y_start = y_stop - (int(y_search_subset/xy_window[1]) * xy_window[0])
    
    window_list = []
    for j in range(y_stop, y_start, -1*int(xy_window[1]*xy_overlap[1])):
        for i in range(x_stop, x_start, -1*int(xy_window[0]*xy_overlap[0])):
            window_list.append(
                (
                    (i-xy_window[0], j-xy_window[1]), (i, j)
                )
            )
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True)['features'])      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)['features']
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
