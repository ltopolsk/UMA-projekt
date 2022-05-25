from tree import Tree
import pandas as pd
import numpy as np
from statistics import mode


class Forest():

    def __init__(self, tr_set: pd.DataFrame, num_trees, num_train_ins, cat_col_name):
        self.tr_set = tr_set
        self.num_trees = num_trees
        self.num_train_ins = num_train_ins
        self.cat_col_name = cat_col_name
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
            self.trees.append(Tree(samples, attributes, self.cat_col_name))

    def classify(self, obj):
        classes = []
        for tree in self.trees:
            classes.append(tree.classify(obj))
        return mode(classes)


if __name__ == "__main__":
    df = pd.read_csv('agaricus-lepiota.data')
    forest = Forest(
                        tr_set=df,
                        num_trees=100,
                        num_train_ins=2000,
                        cat_col_name="cat"
                    )
    print(forest.classify(df.iloc[10]))