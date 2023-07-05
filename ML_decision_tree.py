import pandas as pd
import math


""" 
The entropy function is called by the "infogain" function.  
It takes all the positive ("p") cases and all the  ("e") cases and returns entropy.
For each column in the data.

"""


def entropy(positive_cases, negitive_cases):
    if (positive_cases == 0 or negitive_cases == 0):
        return 0
    sum = positive_cases+negitive_cases
    entropy = ((-positive_cases/sum) * math.log((positive_cases/sum), 2)
               ) - (negitive_cases/sum * math.log((negitive_cases/sum), 2))
    return entropy


""" 
the infogain is called by the "split_decision" function.  
It calulates the information gain for a given row of the dataset. 

"""


def infogain(data, name):
    values_list = data[name].unique()
    values_list = values_list.tolist()
    p_list = [0] * (len(data[name].unique()))
    e_list = [0] * (len(data[name].unique()))

    for index, row in data.iterrows():
        val = values_list.index(row[name])
        if row['label'] == "p":
            p_list[val] = p_list[val]+1
        else:
            e_list[val] = e_list[val]+1
    information_gain = entropy(sum(p_list), sum(e_list))
    count = sum(p_list) + sum(e_list)
    entropy_list = []
    for j in range(len(values_list)):
        entropy_list.append(
            ((p_list[j] + e_list[j]) / count) * entropy(p_list[j], e_list[j]))
    information_gain = information_gain - sum(entropy_list)
    return information_gain


"""
the split_decision takes the data and calculates the split decision based on the information gain.
It returns the 
best infomation gain value in "best_infogai_val"
and the name of the best infomation gain colunn
in "best_infogain_name"

"""


def split_decision(data):

    names = data.columns[1:]
    list_of_infomation_gain = [0] * len(names)
    counter = 0
    for col in names:
        list_of_infomation_gain[counter] = (infogain(data, col))
        counter += 1
    best_infogain_val = max(list_of_infomation_gain)
    best_infogain_name = names[list_of_infomation_gain.index(
        max(list_of_infomation_gain))]

    return best_infogain_val, best_infogain_name


"""
the splitter function takes the data and splits in into sublists
it returns:
sublists
sublists element name

"""


def spliter(data, best_split_name):
    # print(data[best_split_name].unique())
    sublist = []
    for element in data[best_split_name].unique():
        test = []
        for index, row in data.iterrows():
            if row[best_split_name] == element:
                test.append(row)

        test = pd.DataFrame(test)
        del test[best_split_name]
        sublist.append(test)

    return sublist, data[best_split_name].unique()


"""
The perfect_split function takes the data and checks to see if the data was split perfectly 
so that all the labesl for the data is either "p" or "e"

"""


def perfect_split(data):
    return len(data['label'].unique()) == 1


"""
The Node class is where we store both the decision Node and the leaf nodes.
These nodes can be diffrenciated by our program base on the value being stored in condition. 
If it's a tuple then its decision node, else it's a leaf.

There are also some simple functions that enable the readability of the attributes in the node.

"""


class Node:
    def __init__(self, condition, letter, children=[], depth=0):
        self.condition = condition
        self.letter = letter
        self.children = children
        self.depth = depth
        self.correct = 0
        self.incorrect = 0

    def __str__(self):
        return str([self.condition, self.letter, self.children, self.depth])

    def __repr__(self):
        return self.__str__()


"""
The tree_builder function is our recurive function for building the tree.
if we have reached the max depth or our sample data produces a perfect split on that node then it will return a leaf node,
else
It would recursivily create new Nodes that will be fed into the children attribute of the parent node.

"""


def tree_builder(letter, data, max_depth):
    best_split = split_decision(data)
    if max_depth == 0:
        return Node(data["label"].unique()[0]+" "+"depth limit", letter, [], 0)

    if perfect_split(data) == True:
        return Node(data["label"].unique()[0]+" "+"perfect split", letter, [], max_depth)

    lists = spliter(data, best_split[1])
    children = []
    counter = 0
    for list in lists[0]:
        children.append(tree_builder(lists[1][counter], list, max_depth-1))
        counter += 1
    return Node(best_split, letter, children, max_depth)


"""
tree_printer takes the decision tree and prints a visualization of the tree into the terminal. 
each level is denoted by "--------" to visulize the level. each lower level gets more "--------".
It also visulises a given node for the the results of the labels if they were correctly classified or not.

"""


def tree_printer(tree, spacing):
    print(spacing, tree.condition, tree.letter, tree.correct, tree.incorrect)
    for child in tree.children:
        print(spacing, tree_printer(child, spacing+"--------"))


"""
the classifier takes a sample row and evaluates where it will end up in the tree.
it returns True if the label of the sample data and the leaf node align, else false.
this is done recursivly.

"""


def classifier(row, node):
    if type(node.condition) == str:
        if row["label"] == node.condition[0]:
            node.correct += 1
            return True
        else:
            node.incorrect += 1
            return False

    for child in node.children:
        if child.letter == str(row[node.condition[1]]):
            return (classifier(row, child))


"""
the tree_tester takes a Dataframe of testing data and returns 
true positive, false positive, true negitives and false negitive rates.

"""


def tree_tester(tree, test):
    true_positive, true_negitive, false_positive, false_negitive = 0, 0, 0, 0
    test = test.reset_index()

    for index, row in test.iterrows():
        if classifier(row, tree) == True:
            if row["label"] == "p":
                true_positive += 1
            else:
                true_negitive += 1
        else:
            if row["label"] == "p":
                false_positive += 1
            else:
                false_negitive += 1

    return true_positive, true_negitive, false_positive, false_negitive

"""
the metrics_calulator takes a metrics and calculates accuracy, precision and recall.

"""


def metrics_calulator(metrics):
    tp, tn, fp, fn = metrics[0], metrics[1], metrics[2], metrics[3]

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
    Runs Decision Tree algorithm

"""
names = ["label", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", " stalk-surface-above-ring",
         "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
DataFrame = pd.read_csv('agaricus-lepiota.txt', sep=',', names=names)

"""
    Clears all lines of data that include "?"
    

"""

for col in DataFrame.columns:
    DataFrame.drop(DataFrame[DataFrame[col] == "?"].index, inplace=True)




"""
Below runs the program.

"max_depth" determines the max depth of the tree. 

the algorithm is started by the "tree_builder" method which returns the root node of the tree. Which takes:

Letter          -This the letter of the feature being tested as we start, we will call this the root node.

Dataframe       -This is the data that is being used to create the Decision Tree

max_depth       -This is the Max Depth of the tree.  

as inputs. 

and creates a 80/20 split on our data for a train and test subset.
    by randomly sampling data (without replacement) from our Dataset.

"""

df_subset = DataFrame.sample(frac=0.2)  # <--- train
DataFrame = (DataFrame.drop(df_subset.index))  # <--- test

max_depth = 2
tree = tree_builder("root", DataFrame, max_depth)
metrics = (tree_tester(tree, df_subset))
metrics_calulator(metrics)
tree_printer(tree, "")
