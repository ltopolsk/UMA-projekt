import pandas as pd
import numpy as np


class Node():

    def __init__(self, parent=None, children=None, value=None):
        self.parent = parent
        self.children = children
        self.value = value


class Tree():

    def __init__(self, training_set: pd.DataFrame, atributes: pd.Index):
        self.training_set = training_set
        self.atributes = atributes
        self.root = Node()
        self.build_tree()

    def build_tree(self):
        pass

    def _build_tree(self, set: pd.DataFrame, parent: Node):
        # if set is empty:
        # return parent.most_common_class
        if len(set["class"].value_counts()) == 1:
            return set["class"].to_numpy()[0]
        

    def classify(self, object):
        pass

    def _classify(self, object):
        pass

    def _calc_entropy(self, class_col: pd.Series):
        class_counts = class_col.value_counts()
        return np.sum([-value / sum(class_counts) * np.log2(value / sum(class_counts)) for value in class_counts])

    def _calc_inf_gain(self, col: pd.Series, class_col: pd.Series):
        # class_counts = class_col.value_counts()
        col_counts = col.value_counts(sort=False)
        entropy = self._calc_entropy(class_col)
        gain = entropy
        temp_df = pd.concat([class_col, col], axis=1)
        for value, counts in zip(col.unique(), col_counts):
            filtred_class_col = temp_df[temp_df[temp_df.columns[1]] == value][temp_df.columns[0]]
            gain -= counts/sum(col_counts) * self._calc_entropy(filtred_class_col)
        split_info = np.sum([-counts/sum(col_counts) * np.log2(counts/sum(col_counts)) for counts in col_counts])
        return gain/split_info

    def _max_inf_gain(self, df: pd.DataFrame):
        inf_gains = [self._calc_inf_gain(df[column], df["cat"]) for column in df.drop("cat", axis=1).columns]
        value = np.max(inf_gains)
        index = np.argmax(inf_gains)
        return value, index

