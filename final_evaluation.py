from brfss_functions import load_brfss_data, split_brfss_dataset, \
    CustomAttributeDropper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import joblib

data = load_brfss_data()
train, train_labels, test, test_labels = split_brfss_dataset(data)

brfss_classifier = Pipeline([
    ('attr_dropper', CustomAttributeDropper()),
    ('std_scaler', StandardScaler()),
    ('forest_clf', RandomForestClassifier(
        bootstrap=False, n_estimators=400, random_state=42)),
])

brfss_classifier.fit(train, train_labels)

final_predictions = brfss_classifier.predict(test)

test_balanced_accuracy = balanced_accuracy_score(test_labels, final_predictions)
print("Final model balanced accuracy score:\n", test_balanced_accuracy)
# 0.3957300932400573

test_confusion_matrix = confusion_matrix(test_labels, final_predictions)

print("Final model confusion matrix:\n", test_confusion_matrix)
'''
[[40301   129  2311]
 [  783     6   137]
 [ 5342    46  1681]]
'''

joblib.dump(brfss_classifier, "final_brfss_model.pkl")

'''
My final assessment is that this model is not accurate, and is definitely not 
close to being accurate enough to use in a setting that would impact people's
health decisions. It is especially bad at determining if people are pre-diabetic,
most often classifying them as having no diabetes and correctly identifying them
only 6 out of 926 times. Overall it seems as if it is randomly assigning instances
to each class according to how prevelent each class is in the training data. 
I am not sure if I should have performed different cleaning operations on the data, 
but it definitely seems that I should have changed the thresholds for the 
classifiers.
'''