import similarities_for_pipeline as similarity
import data_preprocessing as dp
import pandas as pd
import numpy as np
from torch import Tensor
from pandas import DataFrame
from typing import Any, Dict, List, Set


NUM_SIMILARITIES_PER_UUID = 20


def similarity_matrix(similarity_idx_df: DataFrame, reduced_df: DataFrame) -> DataFrame:
    """
    creates n^2 similarity matrix. Each attraction has a similarity score in relation to each attraction in the data

    Args:
      similarity_idx_df: DataFrame output of the function pairs_df_model
      reduced_df: preprocessed DataFrame

    Returns:
      sqaure DataFrame. columns = index = the indices of the attractions. values: similarity score
    """

    similarity_matrix: DataFrame = pd.DataFrame(
        columns=[i for i in range(reduced_df.shape[0])],
        index=range(reduced_df.shape[0]),
    )
    for i in range(reduced_df.shape[0]):
        for j in range(i, reduced_df.shape[0]):
            if j == i:
                similarity_matrix.iloc[i][j] = 1
                similarity_matrix.iloc[j][i] = 1
            else:
                similarity_score = similarity_idx_df[
                    (similarity_idx_df["ind1"] == i) & (similarity_idx_df["ind2"] == j)
                    ]["score"].values
                similarity_matrix.iloc[i][j] = similarity_score
                similarity_matrix.iloc[j][i] = similarity_score
    return similarity_matrix


def change_idx_and_cols(similarity_matrix: DataFrame, df: DataFrame, col: str) -> DataFrame:
    """
  transform the name of the columns and indices to the name of the specified column

  Args:
    similarity_matrix: square pd.DataFrame of similarity score of each attraction with each attraction
    df: pd.DataFrame of the attractions
    col: The name of the column according to which the columns will be named

  Return:
    list of dictionaries of similarity scores
    """
    similarity_matrix[col] = df[col].apply(lambda uuid: str(uuid))
    similarity_matrix = similarity_matrix.set_index(col)
    similarity_matrix.columns = similarity_matrix.index

    return similarity_matrix


def uuid_similarities(similarity_matrix_df: DataFrame, uuid: str) -> Dict[str, List[str]]:
    """
  extract the specified number of most similar attractions for the uuid

  Args:
    similarity_matrix_df: square pd.DataFrame of similarity score of each attraction with each attraction
    uuid: uuid of the attraction

  Return:
    dictionary of uuid:list[uuid]
    """

    return {
        uuid: list(similarity_matrix_df.loc[uuid].sort_values(ascending=False)[1:NUM_SIMILARITIES_PER_UUID + 1].index)}


def marketplace_similarities(attractions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
  extract the specified number of most similar attractions for each uuid in attractions

  Args:
    attractions: List of dictionaries of the attractions

  Return:
    list of dictionaries of 'uuid':[list of similarities]
    """

    raw_df: DataFrame = pd.DataFrame.from_dict(attractions)
    df_reduced: DataFrame = dp.data_preprocess(raw_df)

    embeddings_text: Tensor = similarity.model_embedding(df_reduced, "text")
    similarity_df: DataFrame = similarity.pairs_df_model(embeddings_text)

    # create a square matrix of the similarity scores
    similarity_matrix_text = similarity_matrix(similarity_df, df_reduced)
    similarity_matrix_uuid = change_idx_and_cols(similarity_matrix_text, df_reduced, "uuid")

    similarities_per_uuid_list = list()
    for uuid in similarity_matrix_uuid.columns:
        similarities_per_uuid_list.append(uuid_similarities(similarity_matrix_uuid, uuid))

    return similarities_per_uuid_list


df = pd.read_csv("sample_berlin.csv")
data = df.to_dict('records')
similarity_json = marketplace_similarities(data)
print(similarity_json)