class ParametersGUI:
    def __init__(self):
        # GUI parameters
        self.learning_rate_options = [
            0.1,
            0.01,
            0.001,
            0.0001,
            0.00001,
            0.000001,
        ]

        self.epoch_min = 5
        self.epoch_max = 100
        self.epoch_step = 5

        # input file
        self.primekg = '~/Desktop/datasets/primekg/kg.csv'

        # form parameters
        self.form_epochs = None
        self.form_lr = None
