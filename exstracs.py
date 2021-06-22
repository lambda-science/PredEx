import warnings

warnings.filterwarnings("ignore")
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
import json
import os

# Load Data Using Pandas
N = [500, 1000, 2000, 3000]
data = pd.read_csv(
    "data/histo_feature.csv"
)  # REPLACE with your own dataset .csv filename
classLabel = "conclusion"
data = data.iloc[:, 9:]
data = data.drop("datetime", axis=1)
data = data.drop(data[data["conclusion"] == "OTHER"].index)
data = data.drop(data[data["conclusion"] == "UNCLEAR"].index)
dataFeatures = data.iloc[
    :, 1:
].values  # DEFINE classLabel variable as the Str at the top of your dataset's action column
dataPhenotypes_raw = data.iloc[:, 0].values
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(dataPhenotypes_raw)
dataPhenotypes = label_encoder.transform(dataPhenotypes_raw)
headers = data.iloc[:, 1:].columns
results_data_plot = {}
# Get Feature Importance Scores to use as Expert Knowledge (see https://github.com/EpistasisLab/scikit-rebate/ for more details on skrebate package)
relieff = ReliefF()
relieff.fit(dataFeatures, dataPhenotypes)
scores = relieff.feature_importances_

for max_N in N:
    # Initialize ExSTraCS Model
    model = ExSTraCS(
        learning_iterations=500000,
        N=max_N,
        expert_knowledge=scores,
        track_accuracy_while_fit=True,
        random_state=777,
    )

    # Model Fit
    trainedModel = model.fit(dataFeatures, dataPhenotypes)
    results_data_plot["name"] = "N" + str(max_N)
    results_data_plot["training_score"] = trainedModel.score(
        dataFeatures, dataPhenotypes
    )
    results_data_plot["instance_coverage"] = trainedModel.get_final_instance_coverage()
    results_data_plot["final_training_acc"] = trainedModel.get_final_training_accuracy()
    results_data_plot[
        "attribute_specificity_list"
    ] = trainedModel.get_final_attribute_specificity_list()
    results_data_plot[
        "attribute_accuracy_list"
    ] = trainedModel.get_final_attribute_accuracy_list()
    results_data_plot[
        "attribute_tracking_sums"
    ] = trainedModel.get_final_attribute_tracking_sums()
    results_data_plot[
        "attribute_coocurrences"
    ] = trainedModel.get_final_attribute_coocurrences(headers)
    instanceLabels = []
    for i in range(dataFeatures.shape[0]):
        instanceLabels.append(i)
    results_data_plot[
        "attribute_tracking_scores"
    ] = trainedModel.get_attribute_tracking_scores(dataPhenotypes)
    # Evalution de la cross-val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    cross_val_scoring = cross_val_score(
        model, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8
    )
    results_data_plot["cross_val_scoring"] = cross_val_scoring

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8
    )
    results_data_plot["train_sizes"] = train_sizes
    results_data_plot["train_scores"] = train_scores
    results_data_plot["test_scores"] = test_scores

    if not os.path.exists(os.path.join("exstracs", results_data_plot["name"])):
        os.makedirs(os.path.join("exstracs", results_data_plot["name"]))

    save_dir = os.path.join("exstracs", results_data_plot["name"])
    # Export Results: training cycle, rule population, model pickle
    with open(os.path.join(save_dir, "results_data_plot.json"), "w") as outfile:
        json.dump(results_data_plot, outfile)
    trainedModel.export_iteration_tracking_data(
        os.path.join(save_dir, "iteration_results.csv")
    )
    trainedModel.export_final_rule_population(
        headers,
        classLabel,
        filename=os.path.join(save_dir, "compacted_rules.csv"),
        RCPopulation=True,
    )
    trainedModel.pickle_model(os.path.join(save_dir, "exstracs_model"))
