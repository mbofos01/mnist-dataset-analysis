from sklearn.svm import SVC  # Example: Support Vector Machine
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
import mnist as mn

data = mn.MNIST()

X = data.digits
y = data.labels

# Create your models
model_svm = SVC(C=10,gamma=0.001,kernel='rbf')  # SVM model
model_rf = skl.LogisticRegression(C=0.01,penalty='l1',solver='liblinear',max_iter=300)  # Random Forest model


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.9, random_state=42)
print("Data Splitted")

# Fit the models
model_svm.fit(train_x, train_y)
print("SVM Fitted")

model_rf.fit(train_x, train_y)
print("Logistic Regression Fitted")

# Make predictions
pred_svm = model_svm.predict(test_x)
print("SVM Predicted")

pred_rf = model_rf.predict(test_x)
print("Logistic Regression Predicted")

# Compute the accuracy scores
acc_svm = model_svm.score(test_x, test_y)
acc_rf = model_rf.score(test_x, test_y)

print(f"SVM Accuracy: {acc_svm}")
print(f"Logistic Regression Accuracy: {acc_rf}")

# see https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/
# for more information about the contingency table

from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

tb_svm = mcnemar_table(y_target=test_y, 
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
