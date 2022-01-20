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


def parse_args(logger):
    """
    This function initialize the parser and the input parameters
    """

    my_parser = argparse.ArgumentParser(description=config_object['params']['description'])
    my_parser.add_argument('--path', '-p',
                           required=True,
                           type=str,
                           help=config_object['params']['path_help'])

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
        # I exclude 'address' from 'drop_duplicates' because in many rows the address is un accurate
        df.drop_duplicates(subset=['name', 'about'], inplace=True)
        df.dropna(subset=['name', 'about', 'address'], inplace=True)
    except KeyError as er:
        logger.debug("One or more columns from the list ['name','about'] are missing from the "
                     "DataFrame!")
        print(er)
        sys.exit(1)

    logger.info(f"Shape after removing duplicates: {df.shape}")
    print("Shape after removing duplicates:", df.shape)
    return df


def model_similarity(text_df, col):
    """
  return ordered similarity df with the columns: index, ind1, ind2, score
  """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Single list of sentences
    sentences = text_df[col].values

    # Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute cosine-similarities for each sentence with each other sentence
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
    and retrieves a similarity df
    """
    df_similarity = model_similarity(df, text_col)
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


def main():
    # config logger
    logger = init_logger()
    logger.info('STARTED RUNNING')

    # parse args
    args = parse_args(logger)

    # validate the input
    if not val_input(args, logger):
        logger.debug('terminating due to invalid input')
        exit()

    # load the data
    raw_df = pd.read_csv(args.path, encoding='UTF-8')

    # 'unavailable' to NAN
    unavailable_to_nan(raw_df)

    # Remove rows which are exactly the same
    df_reduced = remove_duplicates_and_nan(raw_df, logger)

    # Creating similarities DataFrames according to 'name' and 'address'
    name_similarity = df_for_model(df_reduced, "name", "name_score")
    address_similarity = df_for_model(df_reduced, "address", "address_score")

    # Creating similarity DataFrame according to 'about' column and according
    about_similarity = model_similarity(df_reduced, "about")

    # merging all the scores to one dataframe
    similarity_df = merge_df(about_similarity, merge_df(name_similarity, address_similarity))

    # creating a new column of 'final_score' that is the average score of all scores
    scores = [col for col in similarity_df.columns if 'score' in col]
    similarity_df["final_score"] = (similarity_df["score"] + similarity_df["name_score"] + similarity_df[
        "address_score"]) / len(scores)

    # define similarity threshold dataframe
    threshold = 0.8
    similarity_df_threshold = similarity_df[similarity_df["score"] > threshold]

