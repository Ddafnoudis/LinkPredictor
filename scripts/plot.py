import matplotlib.pyplot as plt
from scripts import ResultsGNN


class Plot:
    def __init__(self):
        self.figsize = (12, 8)
        self.title_size = 18
        self.label_size = 16
        self.dpi = 800
    
    def loss_history(self, data: ResultsGNN, outfile: str = None):
        plt.figure(figsize=self.figsize)

        plt.plot(data.n_epochs, data.loss_history, color='blue', label='training')
        plt.plot(data.n_epochs, data.validation_loss_history, color='red', label='validation')

        plt.title('Loss history', fontsize=self.title_size)
        plt.xlabel('loss', fontsize=self.label_size)
        plt.ylabel('epochs', fontsize=self.label_size)
        plt.legend()

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        if outfile:
            plt.savefig(outfile, dpi=self.dpi)
            plt.close()
        else:
            plt.show()
