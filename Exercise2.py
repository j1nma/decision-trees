import numpy as np
import pandas as pd
import os
import subprocess

from sklearn import tree  # tree must have categorical_features and absolute_error implementations

# Dataset dictionaries
golf_nominal = {"dataset_name": "PlayGolf_nominal",
                "label_column": 'Play',
                "label_column_yes": 'yes',
                "label_column_no": 'no',
                "categorical_features": np.arange(0, 9)
                }

golf_numeric = {"dataset_name": "PlayGolf_numeric",
                "label_column": 'Play?',
                "label_column_yes": 'Play',
                "label_column_no": "Don't Play",
                "categorical_features": np.arange(2, 7)
                }

mpg_dict = {"dataset_name": "mpg_dataSet",
            "label_column": 'MpG',
            "label_column_yes": 'good',
            "label_column_no": "bad",
            "categorical_features": np.arange(1, 19)
            }

dictionaries = [golf_nominal, golf_numeric, mpg_dict]

# User's personal id for random seed
my_FH_id = 19034
np.random.seed(my_FH_id)

# DecisionTreeClassifier criteria to be run on, including new 'absolute_error'
criteria = ["gini", "entropy", "absolute_error"]

for dictionary in dictionaries:
    # Create results directory
    dirName = './results-' + dictionary['dataset_name']
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " created.")

    # Read data file
    data_file = os.getcwd() + "/" + dictionary['dataset_name'] + ".csv"
    df = pd.read_csv(data_file)

    # Separate label
    input_samples = df.drop(dictionary['label_column'], axis=1)
    target = df.get(dictionary['label_column'])

    # Encode
    input_samples_encoded = pd.get_dummies(input_samples)
    target = np.array(target.eq(dictionary['label_column_yes']).mul(1))

    for criterion in criteria:
        # Fit
        clf = tree.DecisionTreeClassifier(random_state=my_FH_id, criterion=criterion)
        clf = clf.fit(input_samples_encoded, target, categorical_features=dictionary['categorical_features'])

        # Export tree
        dotPath = dirName + "/" + dictionary['dataset_name'] + '-' + criterion + '.dot'
        tree.export_graphviz(clf, dotPath,
                             feature_names=list(input_samples_encoded.columns),
                             class_names=[dictionary['label_column_no'], dictionary['label_column_yes']])

        # Generate tree PNG
        subprocess.Popen(
            ["dot", "-Tpng", dotPath, "-o", dirName + "/" + dictionary['dataset_name'] + '-' + criterion + ".png"],
            stdout=subprocess.PIPE)
