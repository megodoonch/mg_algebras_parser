"""
Gets the same train/dev/test split as used for the supertagger
and puts them into the same JSON format as the original MGBank files

To run this as a script from the root, you can just change path_to_data to "data"
"""

import json
import os
import re
import shutil
from collections import defaultdict

import nltk

from minimalist_parser.convert_mgbank.mgbank2algebra import clean_original_sentence

# paths
# path_to_data = "../../data/"
# path_to_data = "../data"  # use this if running from scripts
path_to_data = "data"  # use this if running from the root
supertagger_in_path = f"{path_to_data}/raw_data/MGBank/supertagger_data/abstract_data"
path_to_mgbank = f"{path_to_data}/processed/mg_bank"
parent_out_path = f"{path_to_mgbank}/split"
path_to_seed = f"{path_to_data}/raw_data/MGBank/wsj_MGbankSeed"
path_to_auto = f"{path_to_data}/raw_data/MGBank/wsj_MGbankAuto"
path_to_new = f"{path_to_mgbank}/new_parses"
path_to_strings = f"{path_to_data}/raw_data/MGBank/wsj_strings"

# for NLTK tree reading
brackets = '()'
open_b, close_b = brackets
open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
node_pattern = '[^%s%s]+' % (open_pattern, close_pattern)
leaf_pattern = '[^%s%s]+' % (open_pattern, close_pattern)


def directory2sentence_dict(path_to_mgbank, sentence2location, file2sentence):
    """
    Given an MGBank corpus, extract all the sentences and put them and their entry numbers into dicts.
    @param path_to_mgbank: path to seed or auto or new_parses
    @param sentence2location: dict sentence : list of (file, entry number).
    @param file2sentence: file name : list of (sentence, entry number).
    @return: sentence2location, file2sentences
    """
    # Although the sentence sources are given in the file (its name and the entry number tell you its course in the PTB)
    # the sentences in the supertagger set are not always preprocessed the same way as the sentences in the PTB,
    # so we get the output strings from mgbank itself

    subdirectories = os.listdir(path_to_mgbank)
    for directory in subdirectories:
        if os.path.isdir(f"{path_to_mgbank}/{directory}"):
            # loop through MGBank files
            files = os.listdir(f"{path_to_mgbank}/{directory}")
            for file in files:
                if file.endswith(".mrg"):
                    name = file.split(".")[0]
                    # open the file and extract the annotated tree (whose root node has the full string)
                    with open(f"{path_to_mgbank}/{directory}/{file}") as f:
                        data = json.load(f)

                    index_and_tree_pairs = [(entry, data[entry][5]) for entry in data]
                    index_and_sentence_pairs = [(entry, nltk.Tree.fromstring(tree,
                                                                             remove_empty_top_bracketing=True,
                                                                             node_pattern=node_pattern,
                                                                             leaf_pattern=leaf_pattern))
                                                for entry, tree in index_and_tree_pairs]
                    # the entries in this MGBank file
                    index_and_sentence_pairs = [(entry, " ".join(clean_original_sentence(annotated_tree.label()))) for
                                                entry, annotated_tree in index_and_sentence_pairs]

                    for entry, sentence in index_and_sentence_pairs:
                        # print(sentence)
                        # store by lower-cased but keep the original
                        sentence2location[sentence.lower()].append((name, entry, sentence))
                        file2sentence[name].append((sentence, entry))
    return sentence2location, file2sentence


def strings2sentence_dict(path_to_strings_directory, sentence2location, file2sentence):
    """
    Given a directory of just the strings, extract all the sentences and put them and their entry numbers into dicts.
    This doesn't work very well because of differences in pre-processing.
    @param path_to_strings_directory: path to seed or auto or new_parses
    @param sentence2location: dict sentence : list of (file, entry number).
    @param file2sentence: file name : list of (sentence, entry number).
    @return: sentence2location, file2sentences
    """
    subdirectories = os.listdir(path_to_strings_directory)
    for directory in subdirectories:
        if os.path.isdir(f"{path_to_strings_directory}/{directory}"):
            # loop through files
            files = os.listdir(f"{path_to_strings_directory}/{directory}")
            for file in files:
                if file.endswith(".mrg_strings") or file.endswith("sent"):
                    name = file.split(".")[0]
                    # open the file and extract the annotated tree (whose root node has the full string)
                    with open(f"{path_to_strings_directory}/{directory}/{file}") as f:
                        lines = f.readlines()

                    for entry, sentence in enumerate(lines):
                        sentence = sentence.strip()
                        # store by lower-cased but keep the original
                        sentence2location[sentence.lower()].append((name, entry, sentence))
                        file2sentence[name].append((sentence, entry))
    return sentence2location, file2sentence


def find_subcorpus(in_path):
    """
    Given path to supertagging dev/test/train set, find the sentences in the sents_dict we already made
    @param in_path:
    @return: dict of file_name : (entry, sentence) where found; entry is the entry number in the json file
    """
    with open(in_path, 'r') as in_file:
        lines = in_file.readlines()

    sentences = defaultdict(list)
    found = 0
    not_found = []

    lines = [line.strip() for line in lines]
    for line in lines:
        if line.lower() in sents_dict:
            locations = sents_dict[line]
            # some sentences appear more than once; take the top and remove it so next time we take a different one
            file, i, sentence = locations.pop(0)
            # keep the original, un-lower-cased sentence
            sentences[file].append((i, sentence))
            found += 1
        else:
            # print("# not found:", line)
            not_found.append(line)
            continue
    print(f"found {found} out of {len(lines)}")
    return sentences, not_found


def find_corpus_entries(file, file_path, sentences):
    """
    Finds all matching corpus entries for the file
    @param file: mgbank file name
    @param file_path: path to the mgbank file
    @param sentences: dict of file : entry, sentence
    @return: dict of entry: sentence
    """
    found = {}
    with open(file_path) as f:
        data = json.load(f)
        for entry, _ in sentences[file]:
            if str(entry) in data:
                found[entry] = data[str(entry)]
    return found


def find_subcorpus_and_write_to_files(in_path, out_path):
    """
    Reads in file from supertagger data sets (train, test, or dev) and splits the MGBank files and their entries
        into train, dev, and test
    @param in_path: path to supertagger words file
    @param out_path: path to directory to write the new files to
    """
    print("Reading in from", in_path)
    # get the files that match the sentences in the input
    sentences, not_found = find_subcorpus(in_path)
    # if not_found:
    #     print("Not found:")
    # for s in not_found:
    #     print(s)

    all_found = {}

    # loop through dictionary of filename : (sentence, entry)
    # find
    for file in sentences:
        file_name = f"{file}.mrg"

        # the WSJ files (in auto and seed)
        if file.startswith("wsj"):
            directory = file.split("_")[1][:2]
            out_dir = f"{out_path}/{directory}"  # update out path with subdirectory
            # start in auto
            dir_path = f"{path_to_auto}/{directory}/"
            if file_name not in os.listdir(dir_path):
                # try seed
                dir_path = f"{path_to_seed}/{directory}/"

        # the new parses
        else:
            # try new_parses
            dir_path = f"{path_to_new}/new"
            out_dir = f"{out_path}/new"

        # get the corpus entries from the file
        found = find_corpus_entries(file, f"{dir_path}/{file_name}", sentences)

        if len(found) == 0:
            continue
        all_found[file] = [entry for entry in found]
        # write to files if found
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{file}.mrg", 'w') as json_file:
            json.dump(found, json_file)
        with open(f"{out_dir}/{file}.sents", 'w') as sentence_file:
            sentence_file.writelines([f"{entry}: {sentence}\n" for entry, sentence in sentences[file]])

    print("wrote to", out_path)
    return all_found


def wrapper_read_write_subcorpus(name):
    """
    runs the whole shebang on dev, test, or train
    @param name: str, must be dev, test, or train
    """

    assert name in ["dev", "test", "train"], "argument must be dev, test, or train"
    corpus_out_path = f"{parent_out_path}/{name}/"
    corpus_in_path = f"{supertagger_in_path}/{name}.words"

    extracted = find_subcorpus_and_write_to_files(corpus_in_path, corpus_out_path)
    with open(f"{path_to_mgbank}/split/{name}_entries.json", 'w') as f:
        json.dump(extracted, f)
    if name == "train":
        # all the new parses are in the training set
        # hand-fixed missing new parse strings to make them match the inputs
        extracted = find_subcorpus_and_write_to_files(f"{path_to_mgbank}/problematic_new_parse_strings.txt",
                                                      corpus_out_path)
        with open(f"{path_to_mgbank}/split/extra_{name}_entries.json", 'w') as f:
            json.dump(extracted, f)


if __name__ == "__main__":
    print("reading in mgbank")
    sents_dict, files_dict = directory2sentence_dict(path_to_seed, defaultdict(list), defaultdict(list))
    sents_dict, files_dict = directory2sentence_dict(path_to_auto, sents_dict, files_dict)
    sents_dict, files_dict = strings2sentence_dict(path_to_new, sents_dict, files_dict)

    print("\nFinding dev, test, and train")
    wrapper_read_write_subcorpus("dev")
    wrapper_read_write_subcorpus("test")
    wrapper_read_write_subcorpus("train")

    
    # file = "1026"
    # directory = file[:2]
    # in_path = f"{path_to_auto}/{directory}/wsj_{file}.mrg"
    # in_path = f"{path_to_new}/new/new_31.mrg"

    # with open(in_path) as f:
    #     data = json.load(f)

    # print(data["0"][4])
    # nltk.Tree.fromstring(data["0"][5],
    #                      remove_empty_top_bracketing=True,
    #                      node_pattern=node_pattern,
    #                      leaf_pattern=leaf_pattern).draw()

    #
    # copy new parses into numbered files
    # for i, file in enumerate(os.listdir(f"{path_to_seed}/new_parses/")):
    #     # shutil.copyfile(f"{path_to_seed}/new_parses/{file}", f"{parent_out_path}/new_parses/new_{i}.mrg")
    #     shutil.copyfile(f"{path_to_seed}/new_parse_strings/{file}", f"{parent_out_path}/new_parses/new_{i}.sent")
    #

# to_print = 5
# j = 0
# # dict_to_print = dev_sentences
# dict_to_print = sents_dict
# # dict_to_print = files_dict
# while j < to_print:
#     f = list(dict_to_print)[j]
#     print(f, dict_to_print[f])
#     j += 1
