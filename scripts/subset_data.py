"""
A script that takes a subset of the original data and saves it to a new file.

Returns:
    Subset DataFrame with 100 nodes per class.
"""
import pandas as pd
from pandas import DataFrame

class SubsetData:
    def __init__(self, fname: str, outfile: str):
        """
        Initializes the SubsetData class.

        Args:
            fname (str): Path to the input CSV file.
            outfile (str): Path to save the subset CSV file.
        """
        self.fname = fname
        self.outfile = outfile

        # Load the data
        self.df = pd.read_csv(self.fname, sep=',', dtype=object)
        # Types of relation column in the data
        self.relations = self.df["relation"].unique()
        # Print relation types
        self.relation_types()
        # Subset the data by relation column
        self.subset_data()
        # Save the subset data to a folder
        self.save_data()


    def relation_types(self):
        """
        Print the number of unique elements in the relation column.
        """
        print(f"Unique elements before: {len(self.df['relation'].unique())}")


    def subset_data(self):
        """
        Subset the data by relation column, ensuring unique x_name values.
        """
        self.subset_df = pd.DataFrame()

        for relation in self.relations:
            # Get unique x_name values within each relation
            relation_df = self.df[self.df["relation"] == relation].drop_duplicates(subset=["x_name"])
            # Check the number of unique x_name values
            num_unique = len(relation_df)
            # If there are fewer than 100 unique genes, keep all of them
            if num_unique < 100:
                subset_df = relation_df
            else:
                # Otherwise, sample 100 unique x_name values
                subset_df = relation_df.sample(100, random_state=42)
            # Concatenate the subset data
            self.subset_df = pd.concat([self.subset_df, subset_df], ignore_index=True)


    def save_data(self):
        """
        Save the subset data to a folder.
        """
        self.subset_df.to_csv(self.outfile, index=False)


    def get_subset_dataframe(self) -> DataFrame:
        """
        Returns the subset DataFrame containing 100 unique genes per class.

        Returns:
            DataFrame: The subset DataFrame with unique x_name values (if possible).
        """
        return self.subset_df
