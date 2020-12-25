from collections import Counter
from graphviz import Digraph
import random
import numpy as np
import operator



class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot


    # Calculate entropy for a sample
    def calc_entropy(self,node):
        entropy = 0
        sample_size = node['samples']
        for value in node['classCounts'].values():
            entropy -= (value/sample_size)*np.log2(value/sample_size)
        return entropy
    
    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, root):
        max_gain_dict = {}
        for attribute in attributes:
            info_gain = root['entropy']
            for atr_val, attr_index_list in attributes[attribute].items():
                if attr_index_list:
                    sample_targets = {i: target[i] for i in attr_index_list if i in target.keys()}
                    classCount = Counter(sample_targets.values())
                    entropy = 0
                    sample_size = len(sample_targets)

                    for value in classCount.values():
                        entropy -= (value/sample_size)*np.log2(value/sample_size)
                    info_gain-=(sample_size/root['samples'])*entropy
                
            max_gain_dict[attribute] = info_gain

        return max(max_gain_dict.items(), key=operator.itemgetter(1))[0]



    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        data = {i: data[i] for i in range(len(data))}
        target = {i: target[i] for i in range(len(target))}
        for attr_index, attr in enumerate(attributes.keys()):
            attr_vals = attributes[attr]
            index_dict = {attr_val:[] for attr_val in attr_vals}
            for i, sample in data.items():
                index_dict[sample[attr_index]].append(i)
            attributes[attr] = index_dict

        root = self.ID3(data, target, attributes)
        return root
    
    def ID3(self, data, target, attributes, target_Attribute="-", parent_id = -1):
        children_nodes = []
        classCount = Counter(target.values())
        root = self.new_ID3_node()
        root.update({'samples': len(data)})
        root.update({'classCounts': classCount})
        root.update({'entropy': self.calc_entropy(root)})
        root.update({'value':target_Attribute})
        if(not attributes):
            root.update({'label': classCount.most_common()[0][0]})
            self.add_node_to_graph(root, parent_id)
            return root
        elif (len(classCount)==1):
            root.update({'label': classCount.most_common()[0][0]})
            self.add_node_to_graph(root, parent_id)
            return root
        else:

            best_attr = self.find_split_attr(data, target, attributes, root)
            root.update({'attribute': best_attr})
            self.add_node_to_graph(root, parent_id)
            best_attr_dict = attributes[best_attr]
            new_attributes = attributes.copy()
            del new_attributes[best_attr]
            for attr_val, index_list in best_attr_dict.items():
                attr_samples = {i: data[i] for i in index_list if i in data.keys()}
                attr_targets = {i: target[i] for i in index_list if i in target.keys()}

                if attr_samples:
                    node = self.ID3(attr_samples, attr_targets, new_attributes, attr_val, root['id'])
                    children_nodes.append(node)
                    #self.add_node_to_graph(node)
                else:
                    node = self.new_ID3_node()
                    node.update({'label': classCount.most_common()[0][0]})
                    node.update({'samples': 0})
                    node.update({'value':attr_val})
                    #node.update({'classCounts': classCount})
                    self.add_node_to_graph(node, root['id'])
                    children_nodes.append(node)
            
            root.update({'nodes':children_nodes}) 
        return root



    def predict(self, data, tree, attributes):
        attribute_index = {attribute:i for i,attribute in enumerate(attributes.keys())}
        predicted = list()
        for sample in data:
            predicted.append(self.predict_rek(tree, sample, attribute_index))
        return predicted
    
    def predict_rek(self, node, x, attribute_index):
        if node['label'] is not None:
            return node['label']
        else:
            attr = node['attribute']
            attr_ind = attribute_index[attr]
            for nod_child in node['nodes']:
                if(nod_child['value']==x[attr_ind]):
                    return self.predict_rek(nod_child, x, attribute_index)
