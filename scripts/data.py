import torch
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class KnowledgeGraph:
    def __init__(self):
        self.df = None
        self.node_types = {'gene/protein', 'drug', 'effect/phenotype', 'disease'}
        self.node_names = {}
        self.node_names_encoding = {}
        self.edges = defaultdict(list)

    def parse_file(self, fname: str):
        # columns to parse
        my_columns = ['display_relation', 'x_type', 'x_name', 'y_type', 'y_name']

        # parsing file
        df = pd.read_csv(fname, sep=',', usecols=my_columns, dtype=str)

        # filter out nodes that are not of the type of interest
        df = df[df['x_type'].isin(self.node_types)]
        df = df[df['y_type'].isin(self.node_types)]

        self.df = df.reset_index(drop=True)

        # collect further information about the data
        self.preprocess()

    def preprocess(self):
        """
        Collect node names per type and store it in
        a dictionary called `self.node_names`.

        Additionally, create an encoding map {0: name_0, 1: name_1, ...}
        for the nodes of each type and store them under `self.node_names_encoding`

        Finally, creates a dictionary with the triplets as keys and the list of
        of pairs as values, and stores it under `self.edges`
        """
        for node_type in self.node_types:
            sub_df = self.df[self.df['x_type'] == node_type].copy()
            self.node_names[node_type] = sorted(sub_df['x_name'].unique())

        # generate IDs for the node names
        for node_type, node_list in self.node_names.items():
            self.node_names_encoding[node_type] = {
                value: i for i, value in enumerate(node_list)
            }

        # get edgelists
        edge_types = sorted(self.df['display_relation'].unique())
        for edge_type in edge_types:
            sub_df = self.df[self.df['display_relation'] == edge_type].copy()
            for row_list in sub_df.to_records(index=False):
                my_triplet = (row_list[1], row_list[0], row_list[3])
                self.edges[my_triplet].append(
                    (row_list[2], row_list[4])
                )
        self.edges = dict(self.edges)

    def graph(self) -> HeteroData:
        """
        Create heterogeneous graph object from `torch_geometric.data.HeteroData`
        :return: `Heterodata` object representation of the knowledge graph
        """
        # get features for the nodes in the graph
        # TODO: Add the node features from `get_node_features`

        # initialize empty heterogeneous graph
        my_graph = HeteroData()

        # add nodes to the graph
        for node_type, node_list in self.node_names.items():
            node_type = node_type.replace('/', '_')
            my_graph[node_type].node_ids = torch.arange(len(node_list)).to(torch.int64)
            my_graph[node_type].x = torch.ones((len(node_list), 64)).to(torch.float32)

        # encode edges
        encoded_edges = {}
        for edge_type, edge_list in self.edges.items():
            tmp_list = []
            for a, b in edge_list:
                tmp_list.append(
                    (self.node_names_encoding[edge_type[0]][a], self.node_names_encoding[edge_type[2]][b])
                )
            encoded_edges[edge_type] = np.array(tmp_list).T

        # add edges to the graph
        for edge_type, edge_array in encoded_edges.items():
            rel_type = edge_type[1].replace(' ', '_')
            rel_type = rel_type.replace('-', '_')
            edge_type = edge_type[0].replace('/', '_'), rel_type, edge_type[2].replace('/', '_')
            my_graph[edge_type].edge_index = torch.from_numpy(edge_array).to(torch.int64)

        return ToUndirected()(my_graph)

    def get_node_features(self):
        # generate the edge_list
        edge_list = []
        for my_edge_list in self.edges.values():
            for my_edge in my_edge_list:
                edge_list.append(my_edge)
        edge_list = edge_list[:1000]
        # generate networkx graph
        my_graph = nx.Graph()
        my_graph.add_edges_from(edge_list)

        # get Jaccard similarity score
        jaccard_scores = list(nx.jaccard_coefficient(my_graph))
        print(len(jaccard_scores))
