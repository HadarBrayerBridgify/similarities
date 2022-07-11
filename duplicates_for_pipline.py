import uuid
import pandas as pd
import data_preprocessing as dp
from typing import Any, Dict, List


def duplicates(attractions: List[dict[str, str]], similarity: List[dict[str, str]]):
    """
    Generates a duplicate uuid for each attraction with duplicates

    Args:
        attractions: List of dictionaries of the attractions
        similarity: List of dictionaries of uuid: similarity_id

    Returns:
        List of dictionaries, each dictionary contains "uuid" : "duplicate_uuid"
    """

    attractions_df = dp.data_preprocess(pd.DataFrame.from_dict(attractions))
    similarity_df = pd.DataFrame.from_dict(similarity)
    attractions_with_similarities = attractions_df.merge(similarity_df, how="inner", on="title")

    # starting an empty DataFrame for the duplicates
    duplicates_df = pd.DataFrame()

    reliable_sources = ["Musement", "Tiqets", "Ticketmaster", "Getyourguide"]
    not_reliable_suppliers = ["GoogleMaps", "Viator"]

    # checking for duplicates in every group of similarity
    for group in attractions_with_similarities["similarity_uuid"].unique():

        # extracting the rows of a certain group
        similar_group = attractions_with_similarities[attractions_with_similarities["similarity_uuid"] == group]

        # extracting the repeated tags
        tags_count = similar_group["prediction"].value_counts()

        # list of the repeated tags
        repeated_tags = tags_count[tags_count.values > 1].index

        # if there are repeated tags inside the group
        if len(repeated_tags) != 0:

            for tag in repeated_tags:
                # extract the rows from the group that have the same tag:
                similar_tags_rows = similar_group[similar_group["prediction"] == tag]

                # out of the rows with the same tag, extract the rows with the same duration:
                duration_count = similar_tags_rows["duration"].value_counts()
                repeated_duration = duration_count[duration_count.values > 1].index

                # if there are repeated durations in the group
                if len(repeated_duration) != 0:
                    for duration in repeated_duration:
                        similar_duration_and_tags_rows = similar_tags_rows[similar_tags_rows["duration"] == duration]

                        # Find the median
                        median_price = np.median(similar_duration_and_tags_rows["price"])

                        # delete the attractions that are not within 30% of the median
                        price_range = median_price * 0.3
                        bottom_price = median_price - price_range
                        top_price = median_price + price_range
                        idx_to_drop = list()

                        for row in similar_duration_and_tags_rows.index:
                            if not bottom_price <= similar_duration_and_tags_rows.loc[row]["price"] <= top_price:
                                print("bottom:", bottom_price)
                                print("top", top_price)
                                print("price:", similar_duration_and_tags_rows.loc[row]["price"])
                                idx_to_drop.append(row)
                        similar_duration_and_tags_rows.drop(index=idx_to_drop, inplace=True)
                        # if there are no duplicates after dropping the indices
                        if not similar_duration_and_tags_rows.shape[0] > 1:
                            continue

                        # if group source is exclusively from musemunts or tiqets, the attractions may be similars but can't be
                        # duplicates!
                        if len(similar_duration_and_tags_rows["inventory_supplier"].unique()) == 1 and \
                                similar_duration_and_tags_rows["inventory_supplier"].unique()[0] in reliable_sources:
                            continue
                        else:
                            similar_duration_and_tags_rows["duplicate_uuid"] = str(uuid.uuid4())

                            duplicates_df = pd.concat(
                                [duplicates_df, similar_duration_and_tags_rows[["title", "duplicate_uuid"]]])


    # if no duplicates were found
    if duplicates_df.shape[0] == 0:
        print("No duplicates were found!")

    else:
        print("created duplicates")
        duplicates_df.to_dict('records')
        return duplicates_df