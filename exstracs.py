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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# Load Data Using Pandas
# N = [500, 1000, 2000, 3000]
N = [1000]
data = pd.read_csv(
    "data/histo_fake_data.csv"
)  # REPLACE with your own dataset .csv filename
classLabel = "conclusion"
# data = data.iloc[:, 9:]
# data = data.drop("datetime", axis=1)
# data = data.drop(data[data["conclusion"] == "OTHER"].index)
# data = data.drop(data[data["conclusion"] == "UNCLEAR"].index)
dataFeatures = data.iloc[
    :, 1:
].values  # DEFINE classLabel variable as the Str at the top of your dataset's action column
dataPhenotypes_raw = data.iloc[:, 0].values
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(dataPhenotypes_raw)
dataPhenotypes = label_encoder.transform(dataPhenotypes_raw)
headers = data.iloc[:, 1:].columns.to_numpy()
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
    results_data_plot["training_score"] = float(
        trainedModel.score(dataFeatures, dataPhenotypes)
    )
    results_data_plot["instance_coverage"] = trainedModel.get_final_instance_coverage()
    results_data_plot["final_training_acc"] = float(
        trainedModel.get_final_training_accuracy()
    )
    results_data_plot[
        "attribute_specificity_list"
    ] = trainedModel.get_final_attribute_specificity_list()
    results_data_plot[
        "attribute_accuracy_list"
    ] = trainedModel.get_final_attribute_accuracy_list()

    # Evalution de la cross-val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, dataFeatures, dataPhenotypes, cv=cv, scoring="accuracy", n_jobs=8
    )
    results_data_plot["train_sizes"] = train_sizes
    results_data_plot["train_scores"] = train_scores
    results_data_plot["test_scores"] = test_scores

    ######################################################################
    # Target - Data DF
    df = pd.read_csv("data/histo_feature.csv")
    df = df.iloc[:, 9:]
    # Drop les OTHER pour l'instant (que 3 classes)
    df = df.drop(df[df["conclusion"] == "OTHER"].index)
    df = df.drop(df[df["conclusion"] == "UNCLEAR"].index)
    del df["datetime"]
    # Enlever les col remplis de NaN ou avec moins de 5 valeur (annotations)
    df = df.dropna(axis=1, thresh=5)
    df.fillna(0, inplace=True)
    df = df.replace({0.25: 1, 0.5: 1, 0.75: 1})
    # Séparer les features des labels et onehot encoding des labels
    # NM:2, COM:1, UNCLEAR:4, CNM:0, OTHER:3
    X_test, Y_test = df.iloc[:, 1:], df.iloc[:, 0]
    label_encoded_y_test = label_encoder.transform(Y_test)
    #######################################################################
    results_data_plot["y_predict"] = trainedModel.predict(X_test.to_numpy())
    results_data_plot["y_true"] = label_encoded_y_test
    if not os.path.exists(os.path.join("exstracs", results_data_plot["name"])):
        os.makedirs(os.path.join("exstracs", results_data_plot["name"]))

    save_dir = os.path.join("exstracs", results_data_plot["name"])

    # Export Results: training cycle, rule population, model pickle
    with open(os.path.join(save_dir, "results_data_plot.json"), "w") as outfile:
        json.dump(results_data_plot, outfile, cls=NpEncoder, indent=4, sort_keys=True)
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
