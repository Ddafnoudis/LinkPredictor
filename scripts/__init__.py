import torch
import pickle
from typing import List, Dict, Tuple


class EdgeType:
    def __init__(self, src: str, rel: str, dst: str):
        self.edge_type = (src, rel, dst)

    def __call__(self) -> Tuple[str, str, str]:
        return self.edge_type


class EmbeddingSpace:
    def __init__(self, embed_dict: Dict[str, torch.Tensor]):
        self.node_types = sorted(embed_dict.keys())
        self.embedding_dictionary = embed_dict

    def node_types(self) -> List[str]:
        return self.node_types

    def __call__(self) -> Dict[str, torch.Tensor]:
        return self.embedding_dictionary


class ResultsGNN:
    def __init__(self):
        self.n_epochs = None
        self.loss_history = None
        self.validation_loss_history = None
        self.training_embedding_space = None
        self.validation_embedding_space = None
        self.test_embedding_space = None
    
    def set_loss_history(self, history: List[float]):
        self.n_epochs = [i for i in range(1, len(history)+1)]
        self.loss_history = history

    def set_validation_loss_history(self, history: List[float]):
        self.validation_loss_history = history

    def set_training_embeds(self, embs: EmbeddingSpace):
        self.training_embedding_space = embs

    def set_validation_embeds(self, embs: EmbeddingSpace):
        self.validation_embedding_space = embs

    def set_test_embeds(self, embs: EmbeddingSpace):
        self.test_embedding_space = embs

    def save_loss_history(self, output_file: str):
        # combine loss histories
        loss_dict = {
            'training': self.loss_history,
            'validation': self.validation_loss_history,
        }
        with open(output_file, 'wb') as f:
            pickle.dump(loss_dict, f)

    def save_embeddings(
            self,
            training_outfile: str = '',
            validation_outfile: str = '',
            test_outfile: str = ''
    ):
        if training_outfile:
            with open(training_outfile, 'wb') as f:
                pickle.dump(self.training_embedding_space, f)

        if validation_outfile:
            with open(validation_outfile, 'wb') as f:
                pickle.dump(self.validation_embedding_space, f)

        if test_outfile:
            with open(test_outfile, 'wb') as f:
                pickle.dump(self.test_embedding_space, f)
