from sklearn import metrics
from forest import Forest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv('agaricus-lepiota.data')
    df.drop('s_r', axis=1, inplace=True)
    y = df['cat']
    x = df.drop('cat', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.2)
    df1 = x_train.merge(y_train, left_index=True, right_index=True)
    file = open('avaible_values.json')
    availbe_vals = json.load(file)
    forest = Forest(
                        tr_set=df1,
                        num_trees=100,
                        num_train_ins=7000,
                        avaible_vals=availbe_vals,
                        cat_col_name='cat'
                    )
    scores = []
    cats = []
    for _, row in x_test.iterrows():
        cat, score = forest.classify(row)
        scores.append(score)
        cats.append(cat)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label='e')
    print(f"acc={metrics.accuracy_score(y_test, cats)}")
    print(f"precision={metrics.precision_score(y_test, cats, pos_label='e')}")
    print(f"recall={metrics.recall_score(y_test, cats, pos_label='e')}")
    print(f"f1_score={metrics.f1_score(y_test, cats, pos_label='e')}")
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()