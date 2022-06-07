import pandas as pd
import numpy as np
import json


class Node():

    def __init__(self, attribute=None, cat=None, cat_prob=None):
        self.children = []
        self.attribute = attribute
        self.cat = cat
        self.cat_prob = cat_prob

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_cat(self, cat):
        self.cat = cat

    def add_child(self, leading_branch, child, tr_set):
        self.children.append((leading_branch, child, tr_set))

    def set_prob(self, val):
        self.cat_prob = val

    def get_prob(self):
        return self.cat_prob

    def __str__(self) -> str:
        return f"attr={self.attribute}, cat={self.cat}, num_children={len(self.children)}"


class Tree():

    def __init__(self, training_set: pd.DataFrame, attributes: pd.Index, avaible_vals: dict, cat_col_name):
        self.tr_set = training_set
        self.attributes = attributes
        self.avaible_vals = avaible_vals
        self.cat_col_name = cat_col_name
        self.root = Node(cat=training_set[cat_col_name].mode()[0])
        self._set_cat_prob(self.root, self.tr_set)
        self.build_tree()

    def _set_cat_prob(self, node: Node, set: pd.DataFrame):
        values = set[self.cat_col_name].value_counts()
        if 'e' not in values.index:
            node.set_prob(0.0)
            return
        if 'p' not in values.index:
            node.set_prob(1.0)
            return
        node.set_prob(values['e']/(values['e']+values['p']))

    def build_tree(self):
        # print(attributes)
        new_attributes = self._get_split(self.tr_set, self.root, self.attributes)
        for _, new_node, new_tr_set in self.root.children:
            self._build_tree(new_tr_set, new_attributes, new_node)

    def _build_tree(self, tr_set: pd.DataFrame, attributes: pd.Index, cur_node: Node):
        if len(tr_set) == 0:
            return
        if len(tr_set[self.cat_col_name].value_counts()) == 1:
            cur_node.set_cat(tr_set[self.cat_col_name].to_numpy()[0])
            return
        if len(attributes) == 0:
            cur_node.set_cat(tr_set[self.cat_col_name].mode()[0])
            return

        new_attributes = self._get_split(tr_set, cur_node, attributes)
        for _, new_node, new_tr_set in cur_node.children:
            self._build_tree(new_tr_set, new_attributes, new_node)

    def _get_split(self, tr_set: pd.DataFrame, cur_node: Node, attributes: pd.Index) -> pd.Index:
        max_attr = self._max_inf_gain(tr_set, self.cat_col_name, attributes)
        cur_node.set_attribute(max_attr)
        new_attributes = attributes.drop(max_attr)
        for value in self.avaible_vals[max_attr]:
            new_tr_set = tr_set[tr_set[max_attr] == value].drop(max_attr, axis=1)
            new_cat = tr_set[self.cat_col_name].mode()[0] if len(tr_set) != 0 else cur_node.cat
            new_node = Node(cat=new_cat)
            self._set_cat_prob(new_node, new_tr_set)
            cur_node.add_child(value, new_node, new_tr_set)
        return new_attributes

    def classify(self, object: pd.Series):
        return self._classify(object, self.root)

    def _classify(self, object: pd.Series, cur_node: Node):
        if len(cur_node.children) == 0:
            return cur_node.cat, cur_node.get_prob()
        for value, child, _ in cur_node.children:
            if value == object[cur_node.attribute]:
                next_child = child
                break
        return self._classify(object, next_child)

    def print_tree(self):
        self._print_tree(self.root)

    def _print_tree(self, cur_node: Node, depth=0):
        if len(cur_node.children) == 0:
            print(f"{' ' * depth}{cur_node.cat}, {cur_node.get_prob()}")
        else:
            print(f"{' ' * depth}{cur_node.attribute}")
            for attr, child, _ in cur_node.children:
                print(f"{' ' * (depth+1)}{attr}", end=" ")
                self._print_tree(child, depth+1)

    def _calc_entropy(self, class_col: pd.Series):
        class_counts = class_col.value_counts()
        return np.sum([-value / sum(class_counts) * np.log2(value / sum(class_counts)) for value in class_counts])

    def _calc_inf_gain(self, col: pd.Series, class_col: pd.Series):
        col_counts = col.value_counts(sort=False)
        entropy = self._calc_entropy(class_col)
        gain = entropy
        temp_df = pd.concat([class_col, col], axis=1)
        for value, counts in zip(col.unique(), col_counts):
            filtred_class_col = temp_df[temp_df[temp_df.columns[1]] == value][temp_df.columns[0]]
            gain -= counts/sum(col_counts) * self._calc_entropy(filtred_class_col)
        split_info = np.sum([-counts/sum(col_counts) * np.log2(counts/sum(col_counts)) for counts in col_counts])
        return gain/split_info if split_info != 0.0 else gain

    def _max_inf_gain(self, df: pd.DataFrame, cat_col_name, attributes: pd.Index):
        inf_gains = [self._calc_inf_gain(df[column], df[cat_col_name]) for column in attributes]
        index = np.argmax(inf_gains)
        return attributes[index]


if __name__ == "__main__":
    df = pd.read_csv('agaricus-lepiota.data')
    attrs_indexes = [0]+[i for i in range(8, 16)]
    df1 = df.iloc[:, attrs_indexes]
    df1 = df1[0:5000]
    obj = df.iloc[6000]
    file = open('avaible_values.json')
    availbe_vals = json.load(file)
    tree = Tree(
        training_set=df1,
        attributes=df1.columns.drop("cat"),
        cat_col_name="cat",
        avaible_vals=availbe_vals,
    )

    tree.print_tree()
    print(tree.classify(obj), obj["cat"])