import mnist 
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
import pretty_errors

data = mnist.MNIST()
dataframe = data.get_dataframe()
unused = data.get_unused_pixels()
unused_labels = [f"pixel{i-1}" for i in unused]

dataframe_unused = data.get_dataframe()
dataframe_unused.drop(unused_labels, axis=1, inplace=True)

train, test = train_test_split(dataframe, test_size=0.2, random_state=42)        
train_unused, test_unused = train_test_split(dataframe_unused, test_size=0.2, random_state=42)        

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

############################################

# Training labels
y_train_unused = train_unused['label'].values
# Training data (pixels)
x_train_unused = train_unused.drop(labels=['label'], axis=1)
# Testing labels
y_test_unused = test_unused['label'].values
# Testing data (pixels)
x_test_unused = test_unused.drop(labels=['label'], axis=1)

x_train_unused = x_train_unused / 255.0
x_test_unused = x_test_unused / 255.0
print("Data Splitted")




# Create your models
model_lr = skl.LogisticRegression(C=0.01,penalty='l1',solver='liblinear',max_iter=300) 
model_lr_unused = skl.LogisticRegression(C=0.01,penalty='l1',solver='liblinear',max_iter=300) 

# Fit the models
model_lr.fit(x_train, y_train)
print("LR Fitted")

model_lr_unused.fit(x_train_unused, y_train)
print("LR Unused Fitted")

# Make predictions
pred_svm = model_lr.predict(x_test)
print("SVM Predicted")

pred_rf = model_lr_unused.predict(x_test_unused)
print("SVM Unused Predicted")

# Compute the accuracy scores
acc_svm = model_lr.score(x_test, y_test)
acc_unused = model_lr_unused.score(x_test_unused, y_test)

print(f"SVM Accuracy: {acc_svm}")
print(f"SVM Unused Accuracy: {acc_unused}")

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
df_svm = pd.DataFrame(tb_svm, columns=['Correct LR Unused', 'Incorrect LR Unused'], index=['Correct LR', 'Incorrect LR'])

annot_svm = np.array([[val for val in row] for row in tb_svm])
# Plot SVM McNemar Table
plt.figure(figsize=(10,7))
sns.heatmap(df_svm, annot=annot_svm,fmt='d', cmap='YlGnBu')
plt.title('McNemar Table')
plt.savefig('./images/McNemar_Table_SVM.png')
plt.show()

