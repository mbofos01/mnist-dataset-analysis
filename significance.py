from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC  # Example: Support Vector Machine
import sklearn.linear_model as skl
from scipy import stats
import mnist as mn

data = mn.MNIST()

X = data.digits
y = data.labels

# Create a 5x2 cross-validation object
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create your models
model_svm = SVC(C=10,gamma=0.001,kernel='rbf')  # SVM model
model_rf = skl.LogisticRegression(C=0.01,penalty='l1',solver='liblinear',max_iter=300)  # Random Forest model

# Perform cross-validation for SVM
scores_svm = cross_val_score(model_svm, X, y, cv=cv, scoring='accuracy')

print("Done with SVM")
print("SVM Scores: ", scores_svm)

# Perform cross-validation for Random Forest
scores_rf = cross_val_score(model_rf, X, y, cv=cv, scoring='accuracy')

print("Done with Logistic Regression")
print("Logistic Regression Scores: ", scores_rf)

# Compute the average scores for both models
avg_score_svm = scores_svm.mean()
avg_score_rf = scores_rf.mean()

print(f"SVM Average Accuracy: {avg_score_svm}")
print(f"Logistic Regression Accuracy: {avg_score_rf}")

# Perform a paired t-test
t_statistic, p_value = stats.ttest_rel(scores_svm, scores_rf)

print("Threshold: 0.05")
print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")
# Check the p-value
if p_value < 0.05:  # significance level of 0.05
    print("There is a significant difference between the performances of the two models.")
else:
    print("There is no significant difference between the performances of the two models.")
