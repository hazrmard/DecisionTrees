import pandas as pd
import numpy as np
from typing import Tuple, Union, List, Iterable, Dict, Any


class DecisionTree:
    """
    A representation of a Decision Tree for descrete features. Uses pd.DataFrames to process
    instances. Exposes the train() and predict() methods which must be called in that order.
    """
    
    def __init__(self):
        self.root = Node()
    
    
    def train(self, data: pd.DataFrame, label: str, validation: Dict[str, pd.DataFrame] = {}) -> Dict[str, List[float]]:
        """
        Given a set of training instances, creates a decision tree based on the ID3 algorithm.
        Instead of recursive calls, uses a depth first construction based on a stack of features
        to successively split the data on.
        
        Args:
        * data (pd.DataFrame): A DataFrame containing instances where each column is a feature.
            Must contain the target feature/concept as well.
        * label (str): The name of the column containing the target feature/concept to learn.
        * validation (dict): A dictionary mapping the validation set name to a DataFrame of the
            same form as the data argument.
        """
        features = data.columns[data.columns != label]
        values = {feature: data[feature].unique() for feature in features}
        stack = [(self.root, features, data)]
        
        err_hist = {name:[] for name in validation.keys()}

        while len(stack) != 0:
            node, attrs, instances = stack.pop()
            
            # if the current data labels are homogenous, create label
            if self.entropy(instances, label) == 0.0:
                node.leaf = True
                node.label = instances[label].iloc[0]
            
            # if there are no more attributes left to split on, create label based on popularity
            elif len(attrs) == 0:
                node.leaf = True
                node.label = instances[label].mode().iloc[0]
            
            else:
                info_gain, optimal_feature, split = self.max_info_gain(instances, attrs, label)
                node.attribute = optimal_feature
                node.leaf = False

                # for attribute values with matching instances, add them to stack to calculate information
                attrs = attrs[attrs != optimal_feature]
                for feature_value, matching_instances in split:
                    child = Node()
                    child.parent = node
                    node.children[feature_value] = child
                    stack.append((child, attrs, matching_instances))

                # for attribute values with no matching instnces, create label based on popularity
                left_values = values[optimal_feature][~np.isin(values[optimal_feature], list(split.groups.keys()))]
                if len(left_values) > 0:
                    mode = instances[label].mode()[0]
                    for feature_value in left_values:
                        child = Node()
                        child.parent = node
                        child.leaf = True
                        child.label = mode
                        node.children[feature_value] = child
            
            for name, hist in err_hist.items():
                test_data = validation[name]
                error = 1. - (self.predict(test_data) == test_data[label]).sum() / len(test_data)
                hist.append(error)
        
        return err_hist
    
    
    def predict(self, data: pd.DataFrame) -> List[Any]:
        """
        Given a set of instances, computes their labels based on the learned tree.
        
        Args:
        * data (pd.DataFrame): A DataFrame of instances where each column is a feature.
        
        Returns:
        * A list of labels corresponding to each row/instance in the input.
        """
        predictions = []
        for _, row in data.iterrows():
            node = self.root
            while not node.leaf:
                node = node.children[row[node.attribute]]
            predictions.append(node.label)
        return predictions
    
    
    def max_info_gain(self, data: pd.DataFrame, attrs: Iterable[str], label: str) -> Tuple[float, str, pd.core.groupby.GroupBy]:
        """
        Calculates the feature and the information gain of the optimal split.
        
        Args:
        * data (pd.DataFrame): A DataFrame containing instances where each column is a feature.
        * attrs (Iterable[str]): A list of features to compare for information gain.
        * label (str): The feature on which the entropy is calculated.
        
        Returns:
        * information gain (float): The maximum information gain.
        * feature (str): The name of the feature that lead to that gain.
        * sub-groups (pd.core.groupby.GroupBy): References to each subgroup when data is split on that feature.
        """
        gain = -np.inf
        optimal = None
        split = None
        
        N = len(data)
        entropy = self.entropy(data, label)
        
        for attr in attrs:
            attr_gain = entropy        
            groups = data.groupby(attr)
            
            for group_name, group in groups:
                n = len(group)
                attr_gain -= (n / N) * self.entropy(group, label)
            
            if attr_gain > gain:
                gain = attr_gain
                optimal = attr
                split = groups

        return (gain, optimal, split)
    
    
    def entropy(self, data: pd.DataFrame, label: str) -> float:
        """
        Calculates the entropy of a particular feature.
        
        Args:
        * data (pd.DataFrame): A DataFrame containing instances where each column is a feature.
        * label (str): Name of the feature to compute the entropy on. Must be in data.
        
        Returns:
        * entropy (float)
        """
        N = len(data)        
        probs = data[label].value_counts() / N
        entropy = np.sum(-probs * np.log2(probs))
        return entropy


    
class Node:
    """
    Representation of a single point of split in a data set. A node can be either a leaf node
    in which case it has a label. Else it has an attribute on which instances are split for
    each value of that attribute.
    """
    
    def __init__(self):
        self.leaf = True
        self.label = None
        self.attribute = None
        self.parent = None
        self.children = {}  # value -> Node
        self.instances = None
        
        
    def __repr__(self):
        if self.attribute is not None:
            return str(self.attribute) + ' node'
        elif self.leaf:
            return str(self.label) + ' label'