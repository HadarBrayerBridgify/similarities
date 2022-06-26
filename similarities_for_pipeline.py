import re
import uuid
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def unavailable_to_nan(df: pd.DataFrame, col_list: list[str]) -> None:
    """
    change 'unavailable' to empty string in the specified columns

    Args:
      df: raw DataFrame of attractions
      col_list: list of text columns

    Returns:
      None

    """

    for col in col_list:
        df[col] = df[col].apply(lambda x: np.nan if x == 'unavailable' else x)
        df[col] = df[col].fillna("")


def remove_duplicates_and_nan(df: pd.DataFrame) -> None:
    """
    Remove rows which are exactly the same and

    Args:
      df: DataFrame of attractions

    Returns:
      None

     """
    print("Shape before removing duplicates:", df.shape)
    df.drop_duplicates(subset=['title', 'description', 'address'], inplace=True)
    df.dropna(subset=["text"], inplace=True)
    df.reset_index(inplace=True)
    print("Shape after removing duplicates:", df.shape)


def tags_format(df: pd.DataFrame) -> pd.Series:
    """
    Transforming each tag in "categories_list" column to a list of categories

    Args:
      DataFrame of attractions

    Returns:
      a DataFrame column (Series) with a list of categories in each entry
     """

    return df["categories_list"].apply(
        lambda x: list(set([j.strip().title() for j in re.sub(r'[()\[\'"{}\]]', '', x).strip().split(",")])) if type(
            x) != list else x)


def strip_list(df: pd.DataFrame, col: str):
    """
    Remove empty items from a list of each entry of the prediction column

    Args:
      df: DataFrame with a new column for the different tags_format
      col: str, the name of the new column with the new tags_format

    Returns:
      None
    """
    df[col] = df[col].apply(lambda x: [ele for ele in x if ele.strip()])


def model_embedding(df: pd.DataFrame, col: str) -> torch.Tensor:
    """
    calculates the embeddings (as torch) of each entry in 'text' column according to SentenceTransformers

    Args:
      df: preprocessed DataFrame
      col: str, the name of the text column according to which the embeddings will be calculated

    Returns:
      tourch.Tensor
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Single list of sentences
    sentences = df[col].values

    # Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)  # each text transforms to a vector
    print("finished embeddings")
    return embeddings


def data_preprocess(raw_df: pd.DataFrame):
    """
    preprocess the raw DataFrame: update the name of the columns if needed,
    creates 'prediction' column with list of categories,
    creates 'text' column of joining the title and description,
    remove duplicate rows

    Args:
      raw_df: raw DataFrame of attractions

    Returns:
      Pre-processed DataFrame
    """
    raw_df = raw_df.rename(
        columns={"name": "title", "about": "description", "tags": "categories_list", "source": "inventory_supplier",
                 "location_point": "geolocation"})
    if 'prediction' not in raw_df.columns:
        raw_df["prediction"] = tags_format(raw_df)
        strip_list(raw_df, "prediction")
        raw_df["prediction"] = raw_df["prediction"].apply(lambda x: str(x))

    # 'unavailable' to NAN
    unavailable_to_nan(raw_df, ["title", "description"])

    # Remove rows which are exactly the same
    raw_df["text"] = raw_df["title"] + ' ' + raw_df["description"]
    remove_duplicates_and_nan(raw_df)
    print("The data were processed")
    return raw_df


def pairs_df_model(embeddings: torch.Tensor) -> pd.DataFrame:
    """
    Compute cosine-similarities of each embedded vector with each other embedded vector

    Args:
      embeddings: Tensor embeddings of the text column

    Returns:
      DataFrame with columns: 'ind1' (vector index), 'ind2' (vector index), 'score' (cosine score of the vectors)
      (The shape of the DataFrame is: rows: (n!/(n-k)!k!), for k items out of n)

    """
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(cosine_scores) - 1):
        for j in range(i + 1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    # transform to DataFrame and add split the pairs to two colums: 'ind1', 'ind2'
    pairs_df = pd.DataFrame(pairs)

    pairs_df["ind1"] = pairs_df["index"].apply(lambda x: x[0]).values
    pairs_df["ind2"] = pairs_df["index"].apply(lambda x: x[1]).values

    return pairs_df


def similarity_matrix(similarity_idx_df, reduced_df):
    """
    creates n^2 similarity matrix. Each attraction has a similarity score in relation to each attraction in the data

    Args:
      similarity_idx_df: DataFrame output of the function pairs_df_model
      reduced_df: preprocessed DataFrame

    Returns:
      sqaure DataFrame. columns = index = the indices of the attractions. values: simialrity score
    """
    similarity_matrix = pd.DataFrame(columns=[i for i in range(reduced_df.shape[0])], index=range(reduced_df.shape[0]))
    for i in range(reduced_df.shape[0]):
        for j in range(i, reduced_df.shape[0]):
            if j == i:
                similarity_matrix.iloc[i][j] = 1
                similarity_matrix.iloc[j][i] = 1
            else:
                similarity_score = \
                similarity_idx_df[(similarity_idx_df["ind1"] == i) & (similarity_idx_df["ind2"] == j)]["score"].values
                similarity_matrix.iloc[i][j] = similarity_score
                similarity_matrix.iloc[j][i] = similarity_score
    return similarity_matrix


def groups_idx(similarity_df):
    """
    Creates a list of tuples, each tuple is a similarity group which contains the attractions indices (A group consists of the pairs of a particular index and the pairs of
    its pairs. There is no overlap of indices between the groups

    Args:
      similarity_df: DataFrame output of the function pairs_df_model

    Returns:
      a list of tuples. Each tuple contains attractions indices and represent a similarity group
    """
    sets_list = list()

    # go over all the index pairs in the dataframe
    for idx in similarity_df["index"].values:
        was_selected = False

        # list that contains all the groups sets
        first_match = set()

        for group in sets_list:
            # if idx has intersection with one of the groups, add the index to the group
            intersec = set(idx) & group
            if len(intersec) > 0:
                # add the index to the group
                group.update(idx)

                # save in the first group match (and collect if there are more matches)
                first_match.update(group)

                # remove the group (it will be inserted with all the matched items )
                sets_list.remove(group)

                # mark that we have intersection for not adding the idx as different group
                was_selected = True
        # after we iterate over all the groups and found the matches for the idx, insert first_match to the sets_list
        if len(first_match) > 0:
            sets_list.append(first_match)

        if not was_selected:
            sets_list.append(set(idx))

    return sets_list


def groups_df(similarity_df_above_threshold: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """
    Creates a DataFrame of 'uuid' and 'similarity_uuid' of the attractions which have similarity score above the threshold

    Args:
      similarity_df_above_threshold: a filtered DataFrame of the output of pairs_df which pass 'score' > threshold
      df: pre-processed DataFrame of the attractions

    Returns:
      a DataFrame of 'uuid' and 'similarity_uuid'
    """

    # add 'group' column to the above threshold indices and order the dataframe by group
    display_columns = ['title']

    # extract the indices
    above_threshold_idx = list(set(np.array([idx for idx in similarity_df_above_threshold["index"]]).ravel()))

    # extract the relevant rows from the dataframe
    df_above_threshold = df.loc[above_threshold_idx][display_columns]

    df_above_threshold['similarity_uuid'] = 0

    # divide the indices to groups according to similarity
    groups_list = groups_idx(similarity_df_above_threshold)

    # update the group columns according to the groups
    for group in groups_list:
        df_above_threshold['similarity_uuid'].loc[list(group)] = str(uuid.uuid4())

    similarity_groups_json = df_above_threshold.to_dict('records')
    return similarity_groups_json


def main(json_file):

    raw_df = pd.read_json(json_file)

    df_reduced = data_preprocess(raw_df)

    # Creating similarity DataFrame according to 'text'
    embeddings_text = model_embedding(df_reduced, "text")
    embeddings_df = pd.DataFrame(embeddings_text)
    similarity_df = pairs_df_model(embeddings_text)

    # create a square matrix of the similarity scores
    similarity_matrix_text = similarity_matrix(similarity_df, df_reduced)
    similarity_matrix_text.to_csv("similarity_matrix.csv", index=False)

    # filtering according to 'description' column.
    similarity_threshold = 0.78
    similarity_df_above_threshold = similarity_df[similarity_df["score"] > similarity_threshold]

    # extract the rows above the threshold from the dataframe
    similarity_df_json = groups_df(similarity_df_above_threshold, df_reduced)

    return similarity_df_json


if __name__ == "__main__":
    similarity_json = main('SE365 sample.json')