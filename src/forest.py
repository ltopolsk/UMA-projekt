from tree import Tree
import pandas as pd
import numpy as np
from statistics import mode
import json


class Forest():

    def __init__(self, tr_set: pd.DataFrame, num_trees, num_train_ins, cat_col_name, avaible_vals):
        self.tr_set = tr_set
        self.num_trees = num_trees
        self.num_train_ins = num_train_ins
        self.cat_col_name = cat_col_name
        self.avaible_vals = avaible_vals
        self.trees = []
        self.create_forest()

    def create_forest(self):
        for _ in range(self.num_trees):
            samples = self.tr_set.sample(n=self.num_train_ins, replace=True)
            temp_df = self.tr_set.drop(self.cat_col_name, axis=1)
            sampled_temp_df = temp_df.sample(
                                            n=int(np.sqrt(len(temp_df.columns))), 
                                            replace=False, 
                                            axis=1
                                        )
            attributes = sampled_temp_df.columns
            self.trees.append(Tree(samples, attributes, self.avaible_vals, self.cat_col_name))

    def classify(self, obj):
        classes = []
        for tree in self.trees:
            cat, _ = tree.classify(obj)
            classes.append(cat)
        return mode(classes), (classes.count('e')/len(classes)) 


if __name__ == "__main__":
    df = pd.read_csv('agaricus-lepiota.data')
    file = open('avaible_values.json')
    availbe_vals = json.load(file)
    forest = Forest(
                        tr_set=df,
                        num_trees=100,
                        num_train_ins=5000,
                        avaible_vals=availbe_vals,
                        cat_col_name='cat'
                    )
    cat, score = forest.classify(df.iloc[550])
    print(cat)
    print(score)