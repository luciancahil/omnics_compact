import lmdb
import numpy as np
import pickle
import torch
from torch_geometric.nn import GCNConv
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import lmdb
import pickle
from torch_geometric.data import Batch
import torch
import time
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
from torch.nn import Linear, Dropout, Softmax, LeakyReLU
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj, add_self_loops
from torch_geometric.explain import Explainer, GNNExplainer


NAME = "gse129705"
class GCNConvExpose(GCNConv):

    def __init__(self, in_channels, out_channels):
        super().__init__()



    def forward(self, x, cur_edge_index):
        # linear step:
        x_lin = self.lin(x)

        # message passing result:
        mp_out = self.propagate(cur_edge_index, x_lin, None)

        # normally GCNConv adds bias:
        if self.bias is not None:
            out = mp_out + self.bias
        else:
            out = mp_out

        # return BOTH
        return out, mp_out



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

"""
Scale Factor GCN
GCN: Num_nodes to num_graphs. (J x G)
Scale Factor Standard Scaler.
Vanially Neural Network (N x G)
Another Standard Scaler.
One last Neural Network (G x 1).
Return outptut.
"""
class GraphNet(nn.Module):

    def __init__(self, node_scales, num_genes, nodes_per_graph,  graph_scales, graph_means, num_nodes, final_scales, final_means):
        super(GraphNet, self).__init__()  

        self.node_scales = torch.tensor(node_scales).to(torch.float32)
        self.gcn = GCNConv(num_genes, len(nodes_per_graph), normalize=False)

        self.nodes_per_graph = nodes_per_graph
        self.graph_scales = torch.tensor(graph_scales).to(torch.float32)
        self.graph_means = torch.tensor(graph_means).to(torch.float32)
        self.to_graph_layer = nn.Linear(num_nodes, num_graphs)

        self.final_scales = torch.tensor(final_scales).to(torch.float32)
        self.final_means = torch.tensor(final_means).to(torch.float32)
        self.to_final_layer = nn.Linear(num_graphs, 1) 
    
    def forward(self, x, old_edge_index):
        cur_edge_index = add_self_loops(old_edge_index)[0]
        adj = to_dense_adj(old_edge_index)[0].T
        adj = adj.to_sparse()


        x = x.to_dense()
        x = x / self.node_scales
        x  = self.gcn(x, cur_edge_index)



        previous_nodes = 0

        selected_outputs = torch.zeros(x.shape[0])
        for i, num in enumerate(nodes_per_graph):
            start = previous_nodes
            end = previous_nodes + num

            selected_outputs[start:end] = x[start:end,i]


            previous_nodes = end
        
        x = (selected_outputs - self.graph_means) / self.graph_scales

        target = get_saved_info("final_node_logits")[0]


        x = selected_outputs
        
        x = self.to_graph_layer(x)

        x = (x - self.final_means) / self.final_scales

        x = self.to_final_layer(x)

        return sigmoid(x)



        #everything before this is good for example 1. Just resizing, scling, and anns now.



processed_path = os.path.join(".", "data", NAME, "processed")

lmdb_path = os.path.join(processed_path, "Giant_Matrix_Input")

# Data Getters
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

def get_item(idx):
    with env.begin() as txn:
        print("{}".format(idx))
        item_bytes = txn.get("{}".format(idx).encode())

        return pickle.loads(item_bytes)


def get_num_graphs():
    with env.begin() as txn:
        item_bytes = txn.get("num_graphs".encode())

        return pickle.loads(item_bytes)

def get_edge_index():
    with env.begin() as txn:
        item_bytes = txn.get("Graph".encode())

        return pickle.loads(item_bytes)

# get data from a graph.
def get_graph_info(idx):
    X, y = get_item(idx)

    X = X.to(torch.float32)

    return X, y

saved_path = os.path.join(processed_path,  "Saved_Info")
saved_info = lmdb.open(saved_path, readonly=True, lock=False, readahead=False, meminit=False)

def get_saved_info(name):
    with saved_info.begin() as txn:
        item_bytes = txn.get(name.encode())

        return pickle.loads(item_bytes)


# Data Getters



def set_params():
    gn.gcn.lin.weight = nn.Parameter(torch.tensor(get_saved_info("node_coefs")).T.float())
    gn.gcn.bias = nn.Parameter(torch.tensor(get_saved_info("node_intercepts")).float())

    gn.to_graph_layer.weight = nn.Parameter(torch.tensor(get_saved_info("graph_coefs")).T.float())
    gn.to_graph_layer.bias = nn.Parameter(torch.tensor(get_saved_info("graph_intercepts")).float())

    gn.to_final_layer.weight = nn.Parameter(torch.tensor(get_saved_info("final_coefs")).float())
    gn.to_graph_layer.bias = nn.Parameter(torch.tensor(get_saved_info("final_intercept")).float())


"""
    txn.put("num_genes".encode(), pickle.dumps(num_genes))
    txn.put("num_nodes".encode(), pickle.dumps(num_nodes))
    txn.put("nodes_per_graph".encode(), pickle.dumps(nodes_per_graph))
    txn.put("edges_per_graph".encode(), pickle.dumps(edges_per_graph))
    txn.put("genes_per_graph".encode(), pickle.dumps(genes_per_graph))
    txn.put("num_to_name_dicts".encode(), pickle.dumps(num_to_name_dicts))
"""

#total number of genes across everything.
num_genes = get_item("num_genes")

# total number of nodes
num_nodes = get_item("num_nodes")

# list of nodes
nodes_per_graph = get_item("nodes_per_graph")

# list of num edges
edges_per_graph = get_item("edges_per_graph")

# list of num genes
genes_per_graph = get_item("genes_per_graph")

# one dict per grpah.
num_to_name_dicts = get_item("num_to_name_dicts")

# list of all column names.
# will be name of the graph file (with .txt), and undescore, then the gene name
col_names = get_item("col_names")


# the coefficients of the last layer
# will have one value for each graph.
# most will be 0'd out.
final_coefs =  get_saved_info("final_coefs")

# an array, where the ith value is true if the ith final coeficient isn't 0.
live_graphs = np.abs(final_coefs) > 0


# how much to scale each gene.
# we use MaxAbs, so there is no intercept.
gene_scales = get_saved_info("node_transform_scales")

# how much we should scale each node at the second layer.
# we use standard scale, so we need both a scale value and a mean.
node_scales = get_saved_info("graph_transform_scales")

node_means = get_saved_info("graph_transform_means")


#  how much we should scale each number at the last layer, where each value corresponds with a graph.
# we use standard scaler, so we need both scales and means.
graph_scales = get_saved_info("final_transform_scales")
graph_means = get_saved_info("final_transform_means")

nodes_per_graph = get_saved_info("num_nodes")
num_graphs = len(nodes_per_graph)

gn = GraphNet(gene_scales, num_genes, nodes_per_graph, node_scales, node_means,
              num_nodes, graph_scales, graph_means)

edge_index = get_edge_index().long()



# setting pre built data

set_params()


X,y = get_graph_info(0)
pred = gn(X, edge_index)



explainer = Explainer(
    model=gn,
    algorithm=GNNExplainer(epochs=2),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)


explanation = explainer(X.to_dense(), edge_index)

print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")

edge_importance = explanation.edge_mask

breakpoint()
"""
TODO:
- Try filtering: Store Edge num per graph. Then try trimming the graphs that are already 0'd out.
- Also, try filtering genes.
- Turn edge importance into a csv table with rows: Graph, Source, Target, value.
    - This is gonna need the inverted node-to-number map. Store that as well.
- Feature Importance should be graph_name, gene_name, value.
- Rerun this for 200 epochs, and all 128 patients.
    Experiment with epochs. Try 
- Node importance could be either sum of edges involving (to or from) it, or by looking at thing. (Skip For Now).
- Subgraph importance is existing weights.
- Then, write out an explanation.

For now, I want to run the code again, after getting everything.

What do I need?

- Edges per graph.
- number-to-node dict for each graph.
- gene names per graph.



"""