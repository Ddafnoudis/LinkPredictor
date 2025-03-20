import streamlit as st
from scripts.plot import Plot


def main():
    # page header
    st.write('# Loss history')

    # plotting object
    plot = Plot()
    plot.loss_history(
        data=st.session_state.gnn_results,
        outfile='streamlit'
    )



if __name__ == '__main__':
    main()
