import streamlit as st
from scripts import EdgeType
from scripts.nn import run_gnn
from scripts.data import KnowledgeGraph
from scripts.params_gui import ParametersGUI


def main():
    # collect GUI parameters
    params = ParametersGUI()

    # set up page configuration
    st.set_page_config(
        page_title='LinkPredictor',
        layout='wide'
    )

    st.write('# GNN parameters')

    with st.form('my_form'):
        st.write('Parameter set')
        # epochs
        params.form_epochs = st.slider(
            label='epochs',
            min_value=params.epoch_min,
            max_value=params.epoch_max,
            value=params.epoch_min*2,
            step=params.epoch_step
        )

        # learning rate
        params.form_lr = st.select_slider(
            label='learning rate',
            options=params.learning_rate_options,
            value=params.learning_rate_options[1]
        )

        # check for submission
        submitted = st.form_submit_button('Run')
    if submitted:
        run(p=params)

    # sidebar
    st.sidebar.success('This is a demo.')


def run(p: ParametersGUI):
    """
    """
    st.write(f'Epochs: {p.form_epochs}')
    st.write(f'Learning rate: {p.form_lr}')

    # edge of interest
    my_edge_type = EdgeType(src='disease', rel='off_label_use', dst='drug')

    #
    kg = KnowledgeGraph()
    kg.parse_file(fname=p.primekg)

    results = run_gnn(graph=kg.graph(),
                      edge_type=my_edge_type,
                      epochs=p.form_epochs,
                      itta=p.form_lr)


if __name__ == '__main__':
    main()
