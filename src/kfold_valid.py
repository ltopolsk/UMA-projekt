import json
from forest import Forest
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt


def kfold_validate(x: pd.DataFrame, y:pd.Series, hyperparams, avaible_vals, cat_col_name, k):
    models = []
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        df = x_train.merge(y_train, left_index=True, right_index=True)
        model = Forest(
            tr_set=df,
            num_trees=hyperparams[0],
            num_train_ins=hyperparams[1],
            cat_col_name=cat_col_name,
            avaible_vals=avaible_vals
        )
        model_stats = evaluate(model, x_test, y_test)
        models.append(model_stats)
    return models


def evaluate(model: Forest, x_test, y_test):

    scores = []
    cats = []
    for _, row in x_test.iterrows():
        cat, score = model.classify(row)
        scores.append(score)
        cats.append(cat)

    fpr, tpr, _ = metrics.roc_curve(y_test, scores, pos_label='e')
    model_stats = {}
    model_stats["model"] = model
    model_stats["acc"] = metrics.accuracy_score(y_test, cats)
    model_stats["precision"] = metrics.precision_score(y_test, cats, pos_label='e')
    model_stats["recall"] = metrics.recall_score(y_test, cats, pos_label='e')
    model_stats["f1_score"] = metrics.f1_score(y_test, cats, pos_label='e')
    model_stats["auc"] = metrics.auc(fpr, tpr)
    model_stats["fpr"] = fpr
    model_stats["tpr"] = tpr
    return model_stats


def plot_roc(fpr, tpr, auc, fpath):
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], 
             [0, 1], 
             color="navy", 
             lw=2, 
             linestyle="--",
             label="ROC curve (area = %0.3f)" % auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(fpath)


if __name__ == "__main__":
    df = pd.read_csv('../data/agaricus-lepiota.data')
    df.drop(['s_r', 'v_t'], axis=1, inplace=True)
    y = df['cat']
    x = df.drop('cat', axis=1)
    file = open('../data/avaible_values.json')
    availbe_vals = json.load(file)
    num_trees = (10, 50, 1000)
    num_train_ins = (1000, 3000, 5000, 7000)
    for num_tree in num_trees:
        for num_ins in num_train_ins:
            models = kfold_validate(x, y, (num_tree, num_ins), availbe_vals, 'cat', 3)
            for i, model in enumerate(models):
                print(f'Model: {num_tree}_{num_ins}_{i}, Acc:{model["acc"]:.3f}, precision: {model["precision"]:.3f}, auc: {model["auc"]:.3f}')
                plot_roc(model['fpr'], model['tpr'], model['auc'], f"../plots/model_{num_tree}_{num_ins}{i}.png")