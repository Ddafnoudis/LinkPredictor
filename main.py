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
                      epochs=config['epochs'],
                      itta=config['learning_rate'])

    # save to files
    results.save_loss_history(output_file=config['output_dir'] / 'loss_histories.pkl')
    results.save_embeddings(
        training_outfile=config['emb_dir'] / 'training.pkl',
        validation_outfile=config['emb_dir'] / 'validation.pkl',
        test_outfile=config['emb_dir'] / 'test.pkl',
    )

    # plotting
    plot = Plot()
    plot.loss_history(data=results, outfile=config['plot_dir'] / 'loss_history.png')
    plot.embedding_space(data=results, outfile=config['plot_dir'] / 'embeddings.png')


if __name__ == '__main__':
    main()
