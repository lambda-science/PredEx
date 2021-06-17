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
dataFeatures = data.iloc[:,1:].values #DEFINE classLabel variable as the Str at the top of your dataset's action column
dataPhenotypes_raw = data.iloc[:,0].values
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(dataPhenotypes_raw)
dataPhenotypes = label_encoder.transform(dataPhenotypes_raw)
headers = data.iloc[:,1:].columns
#Shuffle Data Before CV
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

#Get Feature Importance Scores to use as Expert Knowledge (see https://github.com/EpistasisLab/scikit-rebate/ for more details on skrebate package)
relieff = ReliefF()
relieff.fit(dataFeatures,dataPhenotypes)
scores = relieff.feature_importances_

#Initialize ExSTraCS Model
model = ExSTraCS(N=3000,expert_knowledge=scores)

trainedModel = model.fit(dataFeatures,dataPhenotypes)
print("Training Score: ", trainedModel.score(dataFeatures,dataPhenotypes))
trainedModel.export_iteration_tracking_data("exstracs/iterationData_postfit.csv")
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/fileRulePopulation_postfit.csv",DCAL=False)
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/popData2_postfit.csv")
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/popData3_postfit.csv",RCPopulation=True)
trainedModel.pickle_model("exstracs/savedModel1_postfit")

# Evalution de la cross-val
# Shuffle Data (useful for cross validation)
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
cross_val_scoring = cross_val_score(trainedModel, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8)
print("Cross-Validation Scores: ", cross_val_scoring)

train_sizes, train_scores, test_scores = learning_curve(model, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
print("Train-Size:", train_sizes)
print("Train-Score: ", train_scores_mean)
print("Test-Score: ", test_scores_mean)

trainedModel.export_iteration_tracking_data("exstracs/iterationData_final.csv")
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/fileRulePopulation_final.csv",DCAL=False)
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/popData2_final.csv")
trainedModel.export_final_rule_population(headers,classLabel,filename="exstracs/popData3_final.csv",RCPopulation=True)
trainedModel.pickle_model("exstracs/savedModel1_final")