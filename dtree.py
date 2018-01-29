import pandas as pd
import numpy as np
from typing import Tuple, Union, List, Iterable


class DecisionTree:
    
    def __init__(self):
        self.root = Node()
    
    
    def train(self, data: pd.DataFrame, label: str):
        """
        Doc string
        """
        features = data.columns[data.columns != label]
        values = {feature: data[feature].unique() for feature in features}
        stack = [(self.root, features, data)]

        while len(stack) != 0:
            node, attrs, instances = stack.pop()
            
            # if the current data labels are homogenous, create label
            if self.entropy(instances, label) == 0.0:
                node.leaf = True
                node.label = instances[label].iloc[0]
                continue
            
            # if there are no more attributes left to split on, create label
            if len(attrs) == 0:
                node.leaf = True
                node.label = instances[label].mode().iloc[0]
                continue

            info_gain, optimal_feature, split = self.max_info_gain(instances, attrs, label)
            node.attribute = optimal_feature
            node.leaf = False
            
            # for attribute values with matching instances, add them to stack
            attrs = attrs[attrs != optimal_feature]
            for feature_value, matching_instances in split:
                child = Node()
                node.children[feature_value] = child
                stack.append((child, attrs, matching_instances))
            
            # for attribute values with no matching instnces, create labels
            left_values = values[optimal_feature][~np.isin(values[optimal_feature], list(split.groups.keys()))]
            if len(left_values) > 0:
                mode = instances[label].mode()[0]
                for feature_value in left_values:
                    child = Node()
                    child.leaf = True
                    child.label = mode
                    node.children[feature_value] = child
    
    
    def predict(self, data: pd.DataFrame):
        pass
    
    
    def max_info_gain(self, data: pd.DataFrame, attrs: Iterable[str], label: str) -> Tuple[float, str, pd.core.groupby.GroupBy]:
        """
        Doc string
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
        N = len(data)        
        probs = data[label].value_counts() / N
        entropy = np.sum(-probs * np.log2(probs))
        return entropy


    
class Node:
    
    def __init__(self):
        self.leaf = True
        self.label = None
        self.attribute = None
        self.children = {}  # value -> Node
        
        
    def __repr__(self):
        if self.attribute is not None:
            return str(self.attribute) + ' node'
        elif self.leaf:
            return str(self.label) + ' label'