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
        stack = [(self.root, features, data)]
        while len(stack) != 0:
            node, attrs, instances = stack.pop()
            
            info_gain, optimal_feature, split = self.max_info_gain(instances, attrs, label)
            node.attribute = optimal_feature
            for feature_value, matching_instances in split:
                child = Node()
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