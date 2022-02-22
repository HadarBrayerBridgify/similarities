# main.py

import os
import sys
import logging
import warnings
import argparse
import datetime
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from configparser import ConfigParser
from sentence_transformers import SentenceTransformer, util

# read config file
config_object = ConfigParser()
config_object.read("pr_config.ini")


def parse_args(logger):
    """
    This function initialize the parser and the input parameters
    """

    my_parser = argparse.ArgumentParser(description=config_object['params']['description'])
    my_parser.add_argument('--path', '-p',
                           required=True,
                           type=str,
                           help="config_object['params']['path_help']")

    my_parser.add_argument('--save', '-s',
                           required=False,
                           type=str, default=None,
                           help=config_object['params']['save_help'])

    args = my_parser.parse_args()
    logger.info('Parsed arguments')
    return args


def val_input(args, logger):
    """
    This function validated that the input file exists and that the output path's folder exists
    """

    if not os.path.isfile(args.path):
        logger.debug('the input file doesn\'t exists')
        return False

    if args.save:
        if '/' in args.save:
            folder = "/".join(args.save.split('/')[:-1])
            if not os.path.exists(folder):
                logger.debug('the output folder doesn\'t exists')
                return False
        else:
            folder = "/".join(args.path.split('/')[:-1])
            args.save = folder + '/' + args.save

    else:
        current_time = datetime.datetime.now()
        save_path = f'processed_data_{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-' \
                    f'{current_time.minute}.xlsx'
        args.save = save_path
        logger.info('the save path was set to default')
    logger.info(f'args={args}')
    logger.info('input was validated')
    return True


def init_logger():
    """
    This function initialize the logger and returns its handle
    """

    log_formatter = logging.Formatter('%(levelname)s-%(asctime)s-FUNC:%(funcName)s-LINE:%(lineno)d-%(message)s')
    logger = logging.getLogger('log')
    logger.setLevel('DEBUG')
    file_handler = logging.FileHandler('pr_log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


def unavailable_to_nan(df):
    """
    Transforming 'unavailable' to np.nan
    """
    text_cols = ["about", "name", "address"]
    for col in text_cols:
        df[col] = df[col].apply(lambda x: np.nan if x == 'unavailable' else x)


def remove_duplicates_and_nan(df, logger):
    """ Remove rows which are exactly the same """

    logger.info(f"Shape before removing duplicates and Nans: {df.shape}")
    print("Shape before removing duplicates and Nans:", df.shape)
    try:
        # I exclude 'address' from 'drop_duplicates' because in many rows the address is inaccurate or missing so the
        # duplicates will be expressed especially according to 'name' and 'about'
        df.drop_duplicates(subset=['name', 'about'], inplace=True)
        df.dropna(subset=['name', 'about', 'address'], inplace=True)
        df.reset_index(inplace=True)
    except KeyError as er:
        logger.debug("One or more columns from the list ['name','about'] are missing from the "
                     "DataFrame!")
        print(er)
        sys.exit(1)

    logger.info(f"Shape after removing duplicates: {df.shape}")
    print("Shape after removing duplicates:", df.shape)
    return df


def model_embedding(text_df, col):
    """
  return the embeddings (as torch) of all the text column
  """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Single list of sentences
    sentences = text_df[col].values

    # Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)  # each text transforms to a vetor
    return embeddings


def pairs_df_model(embeddings):
    """
  receive embeddings as dataframe.
  Return a DataFrame of computed cosine-similarities for each embedded vector with each other embedded vector.
  The shape of the DataFrame supposed to be (len(embedding), len(embeddings))
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
    # pairs_df = pairs_df[pairs_df["score"] > threshold]
    return pairs_df


def df_for_model(df, text_col, name_score):
    """
    The function receives dataframe and a text column (not 'about') according to which the similarity will be calculated
    and retrieves a similarity df with the columns: name_score, "ind1", "ind2"
    """
    embedding = model_embedding(df, text_col)
    df_similarity = pairs_df_model(embedding)
    df_similarity.rename(columns={"score": name_score}, inplace=True)
    return df_similarity.drop(columns=["index"])


def merge_df(df1, df2):
    """
    return merged dataframe according to the values in 'ind1' and ind2'
     """
    return pd.merge(df1, df2, on=["ind1", "ind2"], how="inner")


def groups_idx(similarity_df):  # similarity_df = similarity_df_threshold
    """
    :param similarity_df: receive the similarity df above a certain threshold
    :return: a tuples list of all groups indices (A group consists of the pairs of a particular index and the pairs of
    its couples. There is no overlap of indices between the groups
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


def score_avg(idx_list, pairs_df, score_col):
    """
    return the average similarity score of all combinations in the group
    (May consume a lot of time with big data with many combinations)
    """
    # list for all combinations scores
    score_list = list()

    # all possible combinations from idx_list
    for idx in combinations(idx_list, 2):
        idx = sorted(list(idx))
        if idx in list(pairs_df["index"].values):
            # calculating the average score of the group
            score = pairs_df[score_col][(pairs_df["ind1"] == idx[0]) & (pairs_df["ind2"] == idx[1])].values
            score_list.append(score)
    tensor_avg = (sum(score_list) / len(score_list))[0]
    return tensor_avg.item()


def groups_df(similarity_df_threshold, df):  # df = all_mexico_reduced
    """
    The function receives a similarity df and the city df and return the rows above the threshold from the dataframe
    """

    # add 'group' column to the above threshold indices and order the dataframe by group
    display_columns = ['name', 'address', 'location_point', 'about', 'tags', 'prediction', 'price', 'duration',
                       'source', 'external_id']
    scores = ["score", "name_score", "address_score"]

    # extract the indices
    above_threshold_idx = list(set(np.array([idx for idx in similarity_df_threshold["index"]]).ravel()))

    # extract the relevant rows from the dataframe
    df_above_threshold = df.loc[above_threshold_idx][display_columns]

    # add 'group' column
    df_above_threshold['about_avg_score'] = 0
    df_above_threshold["name_avg_score"] = 0
    df_above_threshold["address_avg_score"] = 0
    df_above_threshold["final_avg_score"] = 0
    df_above_threshold['group'] = 0

    # divide the indices to groups according to similarity
    sets_list = groups_idx(similarity_df_threshold)

    # update the group columns according to the groups
    for i, group in enumerate(sets_list):
        df_above_threshold['group'].loc[list(group)] = i
        df_above_threshold['about_avg_score'].loc[list(group)] = score_avg(list(group), similarity_df_threshold,
                                                                           "score")
        df_above_threshold['name_avg_score'].loc[list(group)] = score_avg(list(group), similarity_df_threshold,
                                                                          "name_score")
        df_above_threshold['address_avg_score'].loc[list(group)] = score_avg(list(group), similarity_df_threshold,
                                                                             "address_score")
        df_above_threshold["final_avg_score"].loc[list(group)] = score_avg(list(group), similarity_df_threshold,
                                                                           "final_score")

    # order the dataframe by 'group'
    df_above_threshold = df_above_threshold.sort_values(by='group')

    # change the order of the columns so that 'group' will be first
    cols = df_above_threshold.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df_above_threshold = df_above_threshold[cols]
    return df_above_threshold


def avg_vec(embeddings_df, group_idx):
    """
    return a np.array of the average vector of the indices of a certain group.
    group_idx can be a set or a list.
    """
    return np.array(np.mean(embeddings_df.loc[group_idx], axis=0))


def group_vectors_df(similarity_df_groups, embeddings):
    """
  creates a DataFrame of group number, and it's average vector
  in order to find similarities between the groups
  """
    groups_vectors_df = pd.DataFrame()
    groups_vectors_df["group"] = similarity_df_groups["group"].unique()
    groups_vectors_df.set_index("group", inplace=True)
    groups_vectors_df["avg_vector"] = 0

    avg_vectors_list = list()
    for group in groups_vectors_df.index:
        # extracting the indices of the group
        group_idx = similarity_df_groups[similarity_df_groups["group"] == group].index
        avg_vectors_list.append(avg_vec(embeddings, group_idx))
    groups_vectors_df["avg_vector"] = avg_vectors_list
    return groups_vectors_df


def embeddings_for_model(group_vectors_df):
    """
    Retrieve vectors dataframe and return the data as a numpy array for the similarity model
    """
    groups_vectors = group_vectors_df["avg_vector"].values
    return np.array(groups_vectors.tolist())


def duplicates(similarity_groups, duplicates_df_groups):
    # remove the duplicates from the similarity_group dataframe
    similarity_groups_no_duplicates = similarity_groups.drop(index=duplicates_df_groups.index)

    # starting an empty DataFrame for the duplicates
    duplicates_df = pd.DataFrame()

    # transform "duration" == np.nan to zero for it to be count in "value_counts()"
    # similarity_groups["duration"] = similarity_groups["duration"].fillna(0)

    # for each group from the similarity DataFrame
    # The below loop ignores attractions with "duration" == NAN and therefore I needed to drop the duplicates
    # from the similarities (most of them has "duration" = NAN) and to concat it again after the loop
    for group in similarity_groups_no_duplicates["group"].unique():

        # extracting the rows of a certain group
        similar_group = similarity_groups_no_duplicates[similarity_groups_no_duplicates["group"] == group]

        # if group source is exclusively from musemunts or tiqets, the attractions may be similars but can't be
        # duplicates!
        if similar_group["source"].unique().all() == "Musement" or similar_group["source"].unique().all() == "Tiqets":
            continue
        # extracting the repeated tags
        tags_count = similar_group["prediction"].value_counts()
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
                        duplicates_df = pd.concat([duplicates_df, similar_duration_and_tags_rows])

    duplicates_df = pd.concat([duplicates_df, duplicates_df_groups])
    if duplicates_df.shape == (0, 0):
        print("No duplicates were found!")
    else:
        return duplicates_df


def data_attributes(df):
    """
    :param df: The groups dataframe that was extracted from the original data
    :return: print the dataframe information
    """
    print("Number of groups in the data:", df["group"].unique())
    print("\n")
    print("Number of rows in the data:", df.shape[0])
    print("\n")
    print("row count in each group:\n")
    print(df.groupby("group")["name"].count())


def main():
    # config logger
    logger = init_logger()
    logger.info('STARTED RUNNING')

    # load the data
    raw_df = pd.read_csv("all_mexico.csv", encoding='UTF-8')

    # 'unavailable' to NAN
    unavailable_to_nan(raw_df)

    # Remove rows which are exactly the same
    df_reduced = remove_duplicates_and_nan(raw_df, logger)

    # Creating similarities DataFrames according to 'name' and 'address'
    name_similarity = df_for_model(df_reduced, "name", "name_score")
    address_similarity = df_for_model(df_reduced, "address", "address_score")

    # Creating similarity DataFrame according to 'about' column and according
    embeddings_about = model_embedding(df_reduced, "about")
    embeddings = pd.DataFrame(embeddings_about)
    about_similarity = pairs_df_model(embeddings_about)

    # merging all the scores to one dataframe
    similarity_df = merge_df(about_similarity, merge_df(name_similarity, address_similarity))

    # creating a new column of 'final_score' that is the average score of all scores
    scores = [col for col in similarity_df.columns if 'score' in col]
    similarity_df["final_score"] = (similarity_df["score"] + similarity_df["name_score"] + similarity_df[
        "address_score"]) / len(scores)

    # filtering according to 'about' column.
    similarity_df_threshold = similarity_df[similarity_df["score"] > float(config_object['const']['threshold'])]

    # extract the rows above the threshold from the dataframe
    similarity_df_groups = groups_df(similarity_df_threshold, df_reduced)

    # saving the groups dataframe to a csv file
    similarity_df_groups.to_csv("results.csv")

    # print groups dataframe information
    data_attributes(similarity_df_groups)

    #### creating dataframe for the similarity between the groups
    groups_vectors = group_vectors_df(similarity_df_groups)

    model_embeddings = embeddings_for_model(groups_vectors)
    groups_similarity = pairs_df_model(model_embeddings)
    groups_similarity.to_csv("similarity_between_groups.csv")

    ### creating a duplicate DataFrame

    # Filtering according to 'about' similarity score > 0.99
    duplicates_threshold = similarity_df_threshold[similarity_df["score"] > 0.99]

    # extract the rows above the threshold 0.99 from the dataframe
    duplicates_df_groups = groups_df(duplicates_threshold, df_reduced)

    duplicates_df = duplicates(similarity_df_groups, duplicates_df_groups)
    duplicates_df.to_csv("duplicates_groups_mexico.csv")


if __name__ == "__main__":
    main()
