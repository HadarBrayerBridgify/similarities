import uuid
import pandas as pd
from pandas import DataFrame
import data_preprocessing as dp
import similarities_for_pipeline as similarity
from typing import Any, Dict, List

NUM_MOST_POPULAR = 20


# catch all attractions by simply using google 'title'
def dict_of_groups(most_popular_google, attractions):
    google_uuids = most_popular_google["uuid"].values
    most_popular_dict = {id: [] for id in google_uuids}
    for i, r in most_popular_google.iterrows():
        for idx, row in attractions.iterrows():
            if r["title"] in row["title"]:
                most_popular_dict[r["uuid"]].append(idx)
    return most_popular_dict


def create_uuid_json(most_popular_dict, attractions):
    most_popular_json = list()
    for google_uuid, idx_group in most_popular_dict.items():
        popular_group_uuid = str(uuid.uuid4())
        if len(idx_group) != 0:
            for idx in idx_group:
                most_popular_json.append(
                    {"attraction_id": str(attractions.loc[idx]["uuid"]), "popular_group_uuid": popular_group_uuid})
        most_popular_json.append({"attraction_id": google_uuid, "popular_group_uuid": popular_group_uuid})
    return most_popular_json


def add_similarity_score(most_popular_dict: Dict, popular_groups_df: DataFrame, similarity_matrix_df: DataFrame) -> None:
    """
    add similarity score column to popular_groups_df

    Args:
      most_popular_dict: dictionary of 'google uuid': [idx], the output of dict_of_groups function
      popular_groups_df: DataFrame of the selected popular attractions and their group uuid
      similarity_matrix_df: Square DataFrame of similarity score of the 'title' of each attraction with each attraction

    Return:
      None
    """
    similarity_scores_list = list()

    for google_uuid in most_popular_dict.keys():
        # find the similarity uuid group
        group_similarity_uuid = popular_groups_df["popular_group_uuid"][
            popular_groups_df["attraction_id"] == google_uuid].values
        # find all the uuid's of the specific group
        uuid_list = popular_groups_df["attraction_id"][
            popular_groups_df["popular_group_uuid"] == group_similarity_uuid[0]].values

        for id in uuid_list:
            score_with_google = similarity_matrix_df.loc[id][google_uuid]
            similarity_scores_list.append(score_with_google)

    popular_groups_df["similarity_score"] = similarity_scores_list
    popular_groups_df["similarity_score"] = popular_groups_df["similarity_score"].astype('float')


def remove_duplicates_id(most_popular_dict, popular_groups_df):
    """
    If attraction_id appears more than once in popular_groups_df, the function leaves the row with the highest similarity score to google attraction

    Args:
      most_popular_dict: dictionary of 'google uuid': [idx], the output of dict_of_groups function
      popular_groups_df: DataFrame of the selected popular attractions and their group uuid (many to many)

    Return:
      DataFrame with one to many, attraction_id and group uuid
    """

    # if id appear more than once, choose popular_group_uuid of the best similarity score
    uuid_counts = popular_groups_df["attraction_id"].value_counts()
    repeated_uuid = uuid_counts[uuid_counts > 1].index

    for id in repeated_uuid:
        similarity_score = 0
        # extract the rows and iterate over it
        id_rows = popular_groups_df[popular_groups_df["attraction_id"] == id]
        for i, row in id_rows.iterrows():
            if row["similarity_score"] > similarity_score:
                similarity_score = row["similarity_score"]
                group_id = row["popular_group_uuid"]

        drop_idx = popular_groups_df[
            (popular_groups_df["attraction_id"] == id) & (popular_groups_df["popular_group_uuid"] != group_id)].index
        popular_groups_df = popular_groups_df.drop(index=drop_idx).reset_index(drop=True)
    return popular_groups_df


def main(attractions):
    city_attractions = pd.DataFrame.from_dict(attractions)
    city_attractions = dp.data_preprocess(city_attractions)

    attractions_google = city_attractions[city_attractions["inventory_supplier"] == "GoogleMaps"]
    most_popular_google = attractions_google.sort_values(by='number_of_reviews', ascending=False)[:NUM_MOST_POPULAR]

    paid_attractions = city_attractions[city_attractions["inventory_supplier"] != "GoogleMaps"]
    popular_groups_dict = dict_of_groups(most_popular_google, paid_attractions)  # {title: [idx]}
    popular_uuid_json = create_uuid_json(popular_groups_dict, paid_attractions)
    popular_groups_df = pd.DataFrame.from_dict(popular_uuid_json)
    selected_attractions = pd.merge(city_attractions, popular_groups_df, how="inner", left_on="uuid",
                                    right_on="attraction_id")
    selected_attractions_json = selected_attractions.to_dict('records')
    similarity_matrix_df = similarity.create_similarity_matrix(selected_attractions_json)
    add_similarity_score(popular_groups_dict, popular_groups_df, similarity_matrix_df)
    popular_groups_df = remove_duplicates_id(popular_groups_dict, popular_groups_df)
    popular_groups_json = popular_groups_df.to_dict('records')
    return popular_groups_json


# testing
city_attractions = pd.read_csv("new_york_data.csv")
# convert to list of dictionaries
attractions = city_attractions.to_dict("records")
popular_groups_json = main(attractions)
print(popular_groups_json)