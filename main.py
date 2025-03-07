from scripts import EdgeType
from scripts.plot import Plot
from scripts.nn import run_gnn
from scripts.utility import Utility
from scripts.data import KnowledgeGraph

CONFIG_FILE = 'config.yaml'


def main():
    # generate class instances
    my_utility = Utility()
    config = my_utility.parse_config(fname=CONFIG_FILE)

    # edge of interest
    my_edge_type = EdgeType(src='disease', rel='off_label_use', dst='drug')

    #
    kg = KnowledgeGraph()
    kg.parse_file(fname=config['knowledge_graph'])

    results = run_gnn(graph=kg.graph(),
                      edge_type=my_edge_type(),
                      epochs=500,
                      itta=0.0001)
    
    # plotting
    plot = Plot()
    plot.loss_history(data=results)



if __name__ == '__main__':
    main()
