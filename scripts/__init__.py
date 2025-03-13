import pickle
from typing import List, Tuple


class EdgeType:
    def __init__(self, src: str, rel: str, dst: str):
        self.edge_type = (src, rel, dst)

    def __call__(self) -> Tuple[str, str, str]:
        return self.edge_type


class ResultsGNN:
    def __init__(self):
        self.n_epochs = None
        self.loss_history = None
        self.validation_loss_history = None
    
    def set_loss_history(self, history: List[float]):
        self.n_epochs = [i for i in range(1, len(history)+1)]
        self.loss_history = history

    def set_validation_loss_history(self, history: List[float]):
        self.validation_loss_history = history

    def save_loss_history(self, output_file: str):
        # combine loss histories
        loss_dict = {
            'training': self.loss_history,
            'validation': self.validation_loss_history,
        }
        with open(output_file, 'wb') as f:
            pickle.dump(loss_dict, f)
