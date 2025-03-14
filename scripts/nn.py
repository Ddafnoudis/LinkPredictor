import tqdm
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GAT, to_hetero
from torch_geometric.transforms import RandomLinkSplit

from typing import Tuple
from scripts import EdgeType, ResultsGNN, EmbeddingSpace
from scripts.loss_function import WeightedBinaryCrossEntropy


class EarlyStop:
    def __init__(self,  patience: int = 10, delta: float = 0.0001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric: float):
        if self.best_score is None:
            self.best_score = metric
        elif metric > self.best_score - self.delta:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0


class LinkPredictor(torch.nn.Module):
    def __init__(
            self,
            edge_type: EdgeType,
            train_graph: HeteroData,
            dropout: float = 0.5,
            depth: int = 3
    ):
        super().__init__()
        # get edge type and size of input features
        self.edge_type = edge_type
        in_features = train_graph[edge_type[0]].x.size(1)

        # initialize GNN encoder
        my_model = GAT(
            in_channels=in_features,
            hidden_channels=in_features*2,
            out_channels=16,
            num_layers=depth,
            #p=dropout,
            add_self_loops=False,
        )

        self.encoder = to_hetero(my_model, metadata=train_graph.metadata())

        # linear classifier
        self.classifier = torch.nn.Linear(
            in_features=32,
            out_features=1,
        )

    def forward(self, graph: HeteroData) -> Tuple[torch.Tensor, EmbeddingSpace]:
        """
        Takes as input a graph and embeds the nodes.
        :param graph: a HeteroData object representing a graph
        :return: a tuple containing the results of prediction and the embedding space itself.
        """
        # generate node embeddings
        embeddings = self.encoder(
            x=graph.x_dict,
            edge_index=graph.edge_index_dict
        )

        # source and destination nodes
        src = graph[self.edge_type].edge_label_index[0]
        dst = graph[self.edge_type].edge_label_index[1]
        edge_predictions = torch.cat(
                tensors=[embeddings[self.edge_type[0]][src], embeddings[self.edge_type[2]][dst]],
                dim=-1
        )
        edge_predictions = self.classifier(edge_predictions)

        return torch.sigmoid(edge_predictions).view(-1), EmbeddingSpace(embed_dict=embeddings)


def run_gnn(
        graph: HeteroData, 
        edge_type: EdgeType, 
        epochs: int = 5, 
        itta: float = 0.01
        ) -> ResultsGNN:
    # get running device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}\n')

    # data split
    transform = RandomLinkSplit(
        num_test=0.1,
        is_undirected=True,
        neg_sampling_ratio=2.0,
        edge_types=[edge_type]
    )
    train_data, val_data, test_data = transform(graph)

    # initialize link predictor
    model = LinkPredictor(
        edge_type=edge_type,
        train_graph=train_data,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=itta)
    criterion = WeightedBinaryCrossEntropy()
    early_stop = EarlyStop()

    # place data to device
    train_data.to(device=device)
    test_data.to(device=device)
    val_data.to(device=device)
    model.to(device=device)

    loss_history, val_loss_history = [], []
    for _ in tqdm.tqdm(range(epochs), desc='Epoch'):
        model.train()
        optimizer.zero_grad()

        y_hat, emb_space_train = model(graph=train_data)
        loss = criterion(y_hat, train_data[edge_type].edge_label)
        loss_history.append(loss.item())
        early_stop(metric=loss.item())

        loss.backward()
        optimizer.step()

        # model validation
        with torch.no_grad():
            model.eval()
            validation_y, emb_space_val = model(graph=val_data)
            validation_loss = criterion(validation_y, val_data[edge_type].edge_label)
            val_loss_history.append(validation_loss.item())

        if early_stop.early_stop:
            break

    # test-set results
    with torch.no_grad():
        model.eval()
        y_test, emb_space_test = model(graph=test_data)

    # results object
    results = ResultsGNN()
    results.set_loss_history(history=loss_history)
    results.set_training_embeds(embs=emb_space_train)
    results.set_validation_embeds(embs=emb_space_val)
    results.set_test_embeds(embs=emb_space_test)
    results.set_validation_loss_history(history=val_loss_history)

    return results
