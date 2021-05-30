# Project---AI-Explainibility
## Resources
- https://keras.io/api/applications/inceptionv3/ :Inception-v3 with Keras
- https://towardsdatascience.com/classify-any-object-using-pre-trained-cnn-model-77437d61e05f :Classification using Inception V3
- https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08 :Explanation behind superpixel Segmentation
- https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.quickshift :Segmentation by quickshift
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html : Linear Regression model

## Motivation for the project
In this project each image is subjected to super-pixel segmentation that divides the image into several sub-regions with similar pixel color and texture characteristics. And then
with help of InceptionV3 model  assign a probability to each superpixel sub-region to belong to one of the 1000 classes that InceptionV3 is originally trained on.

## Procedure 
- Start by reading an image and using the pre-trained InceptionV3 model available in tensorflow to predict the class of such image.
Images
- ![](/image_folder/sampleimg.png)
- Extract the top 5 classes of the image.
  1. Labrador Retriever-0.75885326
  1. American Staffordshire Terrier -0.025036748
	1. Golden Retriever -0.010576082
	1. Bull Mastiff -0.012021404)
	1. Great Dane -0.009483305
- Generate set of superpixels using skimage.segmentation.quickshift() and then find the unique superpixels with the help of numpy.unique()
- ![](/image_folder/segmented.png)
- Generate 150 random perturbations for input image with the help of numpy.random.binomial() .
- ![](/image_folder/pertubated.png)
- Then predict class for every perturbations by  pre-trained classifier.
- Calculate the distances between the generated images and the original image by Sklearn.metrics.pairwise_distances .
- Calculate weights for the perturbations by np.sqrt(np.exp(-distances**2)/(kernel_width)**2))
- Fit a explainable linear regression model using the perturbations, predictions and weights.
- Find the importance factor of each superpixel using the coeff attribute of model.
- The superpixels having greater magnitudes of coefficients would be of more importance. So, sort the coefficients to figure out the top 4 superpixels.
- ![](/image_folder/finaloutput.png)
