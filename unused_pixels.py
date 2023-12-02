import mnist 
from sklearn.svm import SVC  # Example: Support Vector Machine
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split

data = mnist.MNIST()
unused = data.get_unused_pixels()

unused_labels = [f"pixel{i-1}" for i in unused]

dataframe = data.get_dataframe()
dataframe.drop(unused_labels, axis=1, inplace=True)

train, test = train_test_split(dataframe, test_size=0.2, random_state=42)        
# Training labels
y_train = train['label'].values
# Training data (pixels)
x_train = train.drop(labels=['label'], axis=1)
# Testing labels
y_test = test['label'].values
# Testing data (pixels)
x_test = test.drop(labels=['label'], axis=1)

x_train = x_train / 255.0
x_test = x_test / 255.0
print("Data Splitted")


# Create your models
model_svm = SVC(C=10,gamma=0.001,kernel='rbf')  # SVM model
model_rf = skl.LogisticRegression(C=0.01,penalty='l1',solver='liblinear',max_iter=300) 

# Fit the models
model_svm.fit(x_train, y_train)
print("SVM Fitted")

model_rf.fit(x_train, y_train)
print("Logistic Regression Fitted")

# Make predictions
pred_svm = model_svm.predict(x_test)
print("SVM Predicted")

pred_rf = model_rf.predict(x_test)
print("Logistic Regression Predicted")

# Compute the accuracy scores
acc_svm = model_svm.score(x_test, y_test)
acc_rf = model_rf.score(x_test, y_test)

print(f"SVM Accuracy: {acc_svm}")
print(f"Logistic Regression Accuracy: {acc_rf}")

# see https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
# for more information about the contingency table

from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

tb_svm = mcnemar_table(y_target=y_test, 
                   y_model1=pred_svm, 
                   y_model2=pred_rf)

print(tb_svm)
chi2, p = mcnemar(ary=tb_svm, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)

assert( acc_svm == (tb_svm[0][0] + tb_svm[0][1]) / (tb_svm[0][0] + tb_svm[0][1]+ tb_svm[1][0] + tb_svm[1][1])) 

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Convert numpy array to DataFrame for better visualization
df_svm = pd.DataFrame(tb_svm, columns=['Correct LR', 'Incorrect LR'], index=['Correct SVM', 'Incorrect SVM'])

annot_svm = np.array([[val for val in row] for row in tb_svm])
# Plot SVM McNemar Table
plt.figure(figsize=(10,7))
sns.heatmap(df_svm, annot=annot_svm,fmt='d', cmap='YlGnBu')
plt.title('McNemar Table')
plt.savefig('./images/McNemar_Table_SVM.png')
plt.show()

