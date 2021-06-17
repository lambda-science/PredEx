import warnings
warnings.filterwarnings('ignore')
from skExSTraCS import ExSTraCS
from skrebate import ReliefF
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import learning_curve 

#Load Data Using Pandas
data = pd.read_csv("data/histo_feature.csv") #REPLACE with your own dataset .csv filename
classLabel = "conclusion"
data = data.iloc[:,9:]
data = data.drop("datetime",axis=1)
data = data.drop(data[data["conclusion"]=="OTHER"].index)
data = data.drop(data[data["conclusion"]=="UNCLEAR"].index)
dataFeatures = data.iloc[:,1:].values #DEFINE classLabel variable as the Str at the top of your dataset's action column
dataPhenotypes_raw = data.iloc[:,0].values
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(dataPhenotypes_raw)
dataPhenotypes = label_encoder.transform(dataPhenotypes_raw)
headers = data.iloc[:,1:].columns

#Get Feature Importance Scores to use as Expert Knowledge (see https://github.com/EpistasisLab/scikit-rebate/ for more details on skrebate package)
relieff = ReliefF()
relieff.fit(dataFeatures,dataPhenotypes)
scores = relieff.feature_importances_

#Initialize ExSTraCS Model
model = ExSTraCS(learning_iterations=200000,N=3000,expert_knowledge=scores)

# Model Fit
trainedModel = model.fit(dataFeatures,dataPhenotypes)
print("Training Score:\n", trainedModel.score(dataFeatures,dataPhenotypes))

# Evalution de la cross-val
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
cross_val_scoring = cross_val_score(model, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8)
print("Cross-Validation Scores:\n", cross_val_scoring)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(model, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8)
print("Train-Size:\n", train_sizes)
print("Train-Score:\n", train_scores)
print("Test-Score:\n", test_scores)


# Export Results: training cycle, rule population, model pickle
trainedModel.export_iteration_tracking_data("exstracs/iteration_results.csv")
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/compated_rules.csv",RCPopulation=True)
trainedModel.pickle_model("exstracs/exstracs_model.model")