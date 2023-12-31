# Instructions

For this assignment we are going to analyze a data set of handwritten digits. The task is to determine the digit (0..9) that was written, from the pixel image (28x28 greyscale values). The dataset to be used is in the file mnist.csv, and its documentation is in the file mnist-documentation.txt.


The data consists of 42,000 examples where each example has 28x28=784 feature values. The first column contains the class label. To read in the data into Python, type (assuming you have saved the data in "D:/mnist.csv"):
```python
import pandas as pd
mnist_data = pd.read_csv('D:/mnist.csv').values
```

To display an image of a digit, you can use the function "imshow" from the "matlibplot" library. For example, to display an image of the example in the first row of the mnist data set, type:

```python
import matplotlib.pyplot as plt
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28
plt.imshow(digits[0].reshape(img_size, img_size))
plt.show()
```

Perform the following analyses, and present the results in your report:

## Tasks
<ol>

<li> Begin with an exploratory analysis of the data. Can you spot useless variables by looking at their summary statisitcs? Consider the class distribution: what percentage of cases would be classified correctly if we simply predict the majority class?
Report any findings from your exploratory analysis that you think are of interest.
</li>

<li> Derive from the raw pixel data a feature that quantifies "how much ink" a digit costs. Report the average and standard deviation of this feature within each class. If you look at these statistics, can you see which pairs of classes can be distinguished well, and which pairs will be hard to distinguish using this feature?



<strong>Coding example in Python:</strong>
```python
# create ink feature
import numpy as np
ink = np.array([sum(row) for row in digits])
# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]
```
Using only the ink feature, fit a multinomial logit model and evaluate, by looking at the confusion matrix, how well this model can distinguish between the different classes. Since in this part of the assignment we only consider very simple models, you may use the complete data set both for training and evaluation. For example, how well can the model distinguish between the digits "1" and "8"? And how well between "3" and "8"? Scale your feature to have zero mean and unit standard deviation before you fit the multinomial logit model. You can use the function "scale" from "sklearn.preprocessing" for this purpose.

```python
# The reshape is neccesary to call LogisticRegression() with a single feature
from sklearn.preprocessing import scale
ink = scale(ink).reshape(-1, 1)
```
</li>

<li>In addition to "ink", come up with one other feature, and explain why you think it might discriminate well between the digits. Your report should contain an unambiguous description of how the feature is derived from the raw data. Perform the same analysis for this new feature as you did for the ink feature.
Fit a multinomial logit model using both features, and report if and how it improves on the single-feature models.</li>


<li>Fit a multinomial logit model using both features, and report if and how it improves on the single-feature models.
</li>

<li>In this part we use the 784 raw pixel values themselves as features.
Draw a random sample of size 5,000 from the data, and use these 5,000 examples for training and model selection (using cross-validation). Estimate the error of the models finally selected on the remaining 37,000 examples.

You may reduce the level of detail to, for example, a 14x14 pixel image. In that case you will have only 196 features instead of 784. You can use the function "resize" from library "cv2" (OpenCV) for this purpose. If you choose to do so, indicate this clearly in the report.
You should analyse the data 
<ol>
<li>the regularized multinomial logit model (using the LASSO penalty),</li>
<li>support vector machines.</li>
</ol>

<strong>Make sure you use the same sample for training and model selection for each algorithm.
Otherwise the comparison wouldn't be fair!</strong>

For each classification method that you apply, discuss the following:
<ol>
<li>What are the complexity parameters of the classification algorithm, and how did you select their values?</li>

<li>What is the estimated accuracy of the best classifier for this method?</li>
</ol>

For these experiments, you should train models with different parameter settings, and use cross-validation to select the best parameter setting. Use the remaining data to produce an honest estimate of the accuracy of the model finally selected for each method. For example, the complexity parameter for the regularized multinomial logit model with LASSO penalty is "C" (this is the inverse regularization strength, that is, 1/lambda). Try different values of "C" and use cross-validation to pick the value that produces the smallest classification error. Finally, to produce an unbiased estimate of the error for the value of "C" selected, compute its classification error on the remaining data.</li>

<li>Which classification method(s) produced the best classifier for this problem?
After step 5 you will have determined the best parameter settings for each algorithm, and you will have applied the algorithm with these parameter settings to the complete training sample to produce a model. Furthermore you will have estimated the error of these models on the remaining data. Now compare the accuracies of these two models, and perform a statistical test to see if there is a significant difference between their accuracies.</li>

</ol>

## Final Remarks

This assignment is not about producing the best possible predictive model. What counts is the quality of your analysis. The experiments should be performed in a methodologically correct manner, so that your conclusions and findings are reliable. Make sure you describe your analysis in such detail that the reader would be able to reproduce it. Please also pay attention to the quality and readability of your written report. If you have performed an excellent analysis, but you write it down very badly, then you still have a problem. Make sure I enjoy reading your work!

Below you find an overview of relevant algorithms and pointers to their availability in Python.

| Algorithm                      | Library                 | Function(s)                             |
| ------------------------------ | ----------------------- | --------------------------------------- |
| General                        | numpy, pandas           |                                         |
| (regularized) Multinomial Logit| sklearn.linear_model    | LogisticRegression(), LogisticRegressionCV() |
| Support Vector Machine         | sklearn.svm             | SVC()                                   |
| Parameter Tuning               | sklearn.model_selection | GridSearchCV(), RandomizedSearchCV()    |
| Image Display                  | matplotlib.pyplot       | imshow(), show()                        |
| Image Processing               | cv2 (OpenCV)            | resize()                                |
