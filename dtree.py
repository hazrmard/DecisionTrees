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
        stack = [(self.root, features)]
        self.root.instances = data
        self.root.label = self.root.instances[label].mode().iloc[0]
        
        err_hist = {name:[] for name in validation.keys()}

        while len(stack) != 0:
            node, attrs = stack.pop()
            
            # if the current data labels are homogenous, create label
            if self.entropy(node.instances, label) == 0.0:
                node.leaf = True
                node.label = node.instances[label].iloc[0]
            
            # if there are no more attributes left to split on, create label based on popularity
            elif len(attrs) == 0:
                node.leaf = True
                node.label = node.instances[label].mode().iloc[0]
            
            else:
                info_gain, optimal_feature, split = self.max_info_gain(node.instances, attrs, label)
                node.attribute = optimal_feature
                node.leaf = False

                # for attribute values with matching instances, add them to stack to calculate information
                attrs = attrs[attrs != optimal_feature]
                for feature_value, matching_instances in split:
                    child = Node()
                    child.instances = matching_instances
                    child.label = child.instances[label].mode().iloc[0]
                    node.children[feature_value] = child
                    stack.append((child, attrs))

                # for attribute values with no matching instances, create label based on popularity
                left_values = values[optimal_feature][~np.isin(values[optimal_feature], list(split.groups.keys()))]
                if len(left_values) > 0:
                    mode = node.instances[label].mode().iloc[0]
                    for feature_value in left_values:
                        child = Node()
                        child.leaf = True
                        child.label = mode
                        node.children[feature_value] = child
            
            for name, hist in err_hist.items():
                test_data = validation[name]
                error = 1. - (self.predict(test_data) == test_data[label]).sum() / len(test_data)
                hist.append(error)
        
        return err_hist
    
    
    def prune(self, validation: pd.DataFrame, label: str, *data: List[pd.DataFrame]) -> List[List[float]]:
        """
        Iteratively prunes nodes from the decision tree until performance degrades. At each iteration,
        all non-leaf nodes are made leaves one at a time and the validation score recorded. The best
        scoring of those nodes matching or exceeding the prior score is prunedy. The iterations
        repeat until the best performing node cannot match the score of the last iteration's pruned score.
        
        Args:
        * validation (pd.DataFrame): A DataFrame containing instances where each column is a feature.
            Must contain the target feature/concept as well.
        * label (str): The name of the column containing the target feature/concept to learn.
        * data (pd.DataFrame): Any number of DataFrame containing instances where each column is a feature.
            Must contain the target feature/concept as well.
        
        Returns:
        * The error rate histories of the pruned tree on the validation data and other optional data sets.
            Of the form of a list of lists of floats. First list is validation errors. Each list begins
            with the initial error rate before pruning.
        """
        nodes = []           # non-leaf nodes
        stack = [self.root]  # stack of nodes to check
        while len(stack) != 0:
            node = stack.pop()
            if not node.leaf:
                nodes.append(node)
            for _, child in node.children.items():
                stack.append(child)
        
        scores = []
        errors = [[] for _ in data]
        while True:
            # get score before pruning on this iteration
            scores.append((self.predict(validation) == validation[label]).sum())
            pruned_scores = np.zeros(len(nodes))
            # get error rates for optional data on this iteration
            for i, datum in enumerate(data):
                correct = (self.predict(datum) == datum[label]).sum()
                errors[i].append(1. - correct / len(datum))
            # iterate over non-leaf nodes, make them leaf to get pruned scores
            for i, node in enumerate(nodes):
                node.leaf = True
                pruned_scores[i] = (self.predict(validation) == validation[label]).sum()
                node.leaf = False
            # find best pruned score, if matches original score, prune node
            best = np.argmax(pruned_scores)
            if best >= scores[-1]:
                nodes[best].leaf = True
                nodes.pop(best)  
            # if best pruned score cannot match original score, stop
            else:
                # combine validation and other error rates
                errors.insert(0, [1. - s / len(validation) for s in scores])
                return errors
    
    
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
            try:
                while not node.leaf:
                    node = node.children[row[node.attribute]]
            except KeyError:
                # In case the attribute is not present in the tree, the majority
                # label of the parent node is assigned as prediction. This error
                # occurs if the training set does not contain all possible feature
                # values that occur in the test set, particularly when data are
                # discretized
                pass
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
        self.leaf = True        # bool
        self.label = None       # Any
        self.attribute = None   # str
        self.children = {}      # value -> Node
        self.instances = None   # pd.DataFrame
        
        
    def __repr__(self):
        if self.attribute is not None:
            return str(self.attribute) + ' node'
        elif self.leaf:
            return str(self.label) + ' label'