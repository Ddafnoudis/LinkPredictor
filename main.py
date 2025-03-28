import os
from scripts import EdgeType
from scripts.plot import Plot
from scripts.nn import run_gnn
from scripts.utility import Utility
from scripts.data import KnowledgeGraph
from scripts.subset_data import SubsetData

CONFIG_FILE = 'config.yaml'




def main():
    # generate class instances
    my_utility = Utility()
    config = my_utility.parse_config(fname=CONFIG_FILE)

    
    # Initialize the SubsetData class
    subset_data = SubsetData(fname=config['knowledge_graph'], outfile=config["subset_file"])
    # Get the subset DataFrame
    subset_df = subset_data.get_subset_dataframe()
    # Print the first 5 rows of the DataFrame
    print(f"Unique elements after: {len(subset_df['relation'].unique())}")
    # exit()

    # edge of interest
    my_edge_type = EdgeType(src='disease', rel='off_label_use', dst='drug')

    # generate graph
    kg = KnowledgeGraph()
    kg.parse_file(fname=config['knowledge_graph'])

    # initialize GNN
    results = run_gnn(graph=kg.graph(),
                      edge_type=my_edge_type,
                      epochs=config['epochs'],
                      itta=config['learning_rate'],
                      classification_type=config['classifier'])

    # annotate & save results
    results.annotate_results(kg=kg)
    results.save_results(output_file=config['output_dir'] / 'predictions.tsv')

    # save to files
    results.save_loss_history(output_file=config['output_dir'] / 'loss_histories.pkl')
    results.save_embeddings(
        training_outfile=config['emb_dir'] / 'training.pkl',
        validation_outfile=config['emb_dir'] / 'validation.pkl',
        test_outfile=config['emb_dir'] / 'test.pkl',
    )

    exit()
    # plotting
    plot = Plot()
    plot.loss_history(data=results, outfile=config['plot_dir'] / 'loss_history.png')
    plot.embedding_space(data=results, outfile=config['plot_dir'] / 'embeddings.png')


if __name__ == '__main__':
    main()
