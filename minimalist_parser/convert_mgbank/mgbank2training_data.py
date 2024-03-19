"""
transforms MGBank into seq2seq training raw_data

Usage:
PYTHONPATH=./ python convert_mgbank/mgbank2training_data.py path/to/mgbank/corpus/directory/ output/directory output_filename (will be given .term suffix)

general e.g.
PYTHONPATH=./ python convert_mgbank/mgbank2training_data.py corpora/MGBank/wsj_MGbank{Auto|Seed}/ corpora/seq2seq/ {seed|auto}
specific e.g.
PYTHONPATH=./ python convert_mgbank/mgbank2training_data.py corpora/MGBank/wsj_MGbankSeed/ corpora/seq2seq/ seed
"""
import logging
import os
import re #newer
import sys
import json
from collections import defaultdict
import nltk
import random

from minimalist_parser.algebras.algebra import AlgebraTerm
# my modules
from minimalist_parser.algebras.hm_interval_pair_algebra import HMIntervalPairsAlgebra
from minimalist_parser.minimalism.mgbank_input_codec import read_corpus_file, tree2term
from minimalist_parser.analysis.term_output_codec import term2nltk_tree
from minimalist_parser.convert_mgbank.mgbank2algebra import nltk_trees2list_term, nltk_trees2string_list_term
from minimalist_parser.minimalism.prepare_packages.interval_prepare_package import IntervalPairPrepare
from minimalist_parser.minimalism.prepare_packages.prepare_packages_hm import PreparePackagesHM
from minimalist_parser.trees.transducer import nltk2tree
from minimalist_parser.minimalism.movers import ListMovers
from minimalist_parser.minimalism.mg_errors import SMCViolation

# VERBOSE = True
VERBOSE = False


def log(s):
    if VERBOSE:
        print(s)


def directory2intervals_file(corpus_path, output_path):
    """
    Wrapper function.
    given a directory of MGbank files, converts them to minimalist algebra terms
        over pairs of intervals and ListMovers mover type
    @param corpus_path: str: path to MGbank corpus; should have subfolders
    @param output_path: str: path to directory to print json outputs to
    """
    print(f"transforming corpus in {corpus_path}...")
    output_errors = []
    corpus_size = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for directory in os.listdir(corpus_path):
        if os.path.isdir(f"{corpus_path}/{directory}"):
            for f in os.listdir(f"{corpus_path}/{directory}"):
                if f.endswith(".mrg"):
                    log(f"\nFile: {f}")
                    parts = f.split(".")
                    out_name = f"{parts[0]}.terms"
                    to_print = {}
                    tree_pairs = read_corpus_file(f"{corpus_path}/{directory}/{f}")

                    for i, (p, a) in enumerate(tree_pairs):
                        try:
                            # get the terms
                            outcome = nltk_trees2list_term(p, a, f, i)
                            # convert back to nltk trees so I can use their input codec
                            # margin is huge because it seems to stop it from making carriage returns in the file
                            to_print[i] = term2nltk_tree(outcome).pformat(
                                margin=100000000000, indent=0, nodesep="",
                                parens="[]", quotes=False)
                        except Exception as error:
                            output_errors.append((f, i, error))
                        corpus_size += 1
                    # print("to_print:", to_print)
                    with open(f"{output_path}/{out_name}", 'w') as out_f:
                        out_f.write(json.dumps(to_print))

    print("Errors during transformation:")
    smc = 0
    for error in output_errors:
        if type(error[2]) != SMCViolation:
            print(error)
        else:
            smc += 1
            log(error)
    print(f"\nTotal errors in transformation: {len(output_errors)}/{corpus_size}")
    print(f"\tof which {smc} are SMC violations")
    print(f"printed terms to {output_path}")


def directory2strings_file(corpus_path, output_path, errors_ok=False):
    """
    Wrapper function.
    Given a directory of MGbank files, converts them to minimalist algebra terms
    over string triples and ListMovers mover type using `nltk_trees2string_list_term`.

    Prints them all to one TSV file with sentence \t tree.

    Creates training data for seq2seq models that output the tree as a string.

    Each opening and closing bracket is its own token, and each op name is its own word.
    @param corpus_path: str: path to MGbank corpus; should have subfolders
    @param output_path: str: path to file to print raw_data to
    """
    
    print(f"transforming corpus in {corpus_path}...")
    output_errors = []
    corpus_size = 0

    # make path if needed
    path_parts = output_path.split('/')
    dir_path = '/'.join(path_parts[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(output_path, 'w') as out_f:
        for directory in os.listdir(corpus_path):
            if os.path.isdir(f"{corpus_path}/{directory}"):
                for f in os.listdir(f"{corpus_path}/{directory}"):
                    # if f == "new_31.mrg":
                    #     print("\n**** new 31 ***")
                    if f.endswith(".mrg"):
                        corpus_size = write_terms_to_tsv_from_mgbank_file(corpus_path, corpus_size, directory, f, out_f,
                                                                          output_errors, errors_ok=errors_ok)

    print("Errors during transformation:")
    smc = 0
    for error in output_errors:
        if type(error[2]) != SMCViolation:
            print(error)
        else:
            smc += 1
            log(error)
    print(f"\nTotal errors in transformation: {len(output_errors)}/{corpus_size}")
    print(f"\tof which {smc} are SMC violations")
    print(f"printed data to {output_path}")


def write_terms_to_tsv_from_mgbank_file(corpus_path, corpus_size, directory, f, out_f, output_errors,
                                        sentence_number=None, errors_ok=False):
    """
    Wrapper function that reads in a single MGBank corpus file, transforms all trees within, and writes them to TSV,
        sentence \t tree as string
    @param corpus_path: path to MGBank corpus
    @param corpus_size: current corpus size as we work our way through the MGBank files
    @param directory: input directory
    @param f: input file within directory
    @param out_f: output file
    @param output_errors: ongoing list of errors, (file name, sentence index, error)
    @param sentence_number: if None, all in file, else just this one
    @param errors_ok: if True, ignore errors if possible in transforming tree. Some output trees will be uninterpretable
    @return: new corpus size.
    """
    log(f"\nFile: {f}")
    tree_pairs = read_corpus_file(f"{corpus_path}/{directory}/{f}")
    for i, (p, a) in enumerate(tree_pairs):
        if sentence_number is not None and i != sentence_number:
            continue
        try:
            generate_term_and_write_to_file(a, p, f, i, out_f, errors_ok=errors_ok)
            corpus_size += 1
        except Exception as error:
            output_errors.append((f, i, error))
            # raise error
    if corpus_size == 0:
        print(f"Warning: no terms created; probably there was no {sentence_number}th entry in the file")
    return corpus_size


def one_mgbank_entry2tsv(corpus_path, in_sub_dir, in_file, out_path, sentence_number=None, errors_ok=False):
    write_terms_to_tsv_from_mgbank_file(corpus_path, 0, in_sub_dir, in_file, out_path, [],
                                        sentence_number=sentence_number, errors_ok=errors_ok)


def generate_term_and_write_to_file(annotated, plain, in_file, i, out_f, errors_ok=False):
    # get the terms
    outcome, sentence = nltk_trees2string_list_term(plain, annotated, in_file, i, errors_ok=errors_ok)
    # convert back to nltk trees so I can use their input codec
    # margin is huge because it seems to stop it from making carriage returns in the file
    tree = term2nltk_tree(outcome).pformat(
        margin=100000000000, indent=0, nodesep="",
        parens="()", quotes=False)
    tree = tree.replace("(", "( ")  # make spaces between all parens
    tree = tree.replace(")", ") ")
    
    #Newer
    sh_amount = 0
    sh_del = 0
    sh_number = 0

    algoritm_type = get_setting("Deletion_Option")
    if algoritm_type == -3:
        # Count the number of SH's of the tree inside the deletion set, divided by total amount of SH's
        sh_perc = calc_sh_del_perc(tree, get_setting("Deletion_Set"))
    else:
        sh_perc = 0

    tree = remove_silent_heads(tree, algoritm_type, sh_perc)
    
    out_f.write(f"{sentence}\t{tree}\n")  # print sentence TAB tree, one per line
    # with open("data/processed/seq2seq/official/test/fake_predictions.txt", 'a') as preds:
    #     preds.write(tree + "\n")
    log(f"{in_file} {i}: {sentence}")


#NEWer
def remove_silent_heads(tree, algorithm_type, sh_perc):
    #tree always begins with (
    if tree[1:8] in [" merge1", " merge2"]:
        index_parenthesis = tree.index('(', 2)
        arg1 = getarg(tree[index_parenthesis:])
        arg2 = getarg(tree[index_parenthesis+len(arg1)+2:])
        retval1 = remove_silent_heads(arg1, algorithm_type, sh_perc)
        retval2 = remove_silent_heads(arg2, algorithm_type, sh_perc)
        if retval1 == "_sh_" and retval2 == "_sh_":
            return "_sh_"
        elif retval1 == "_sh_":
            return retval2
        elif retval2 == "_sh_":
            return retval1
        else:
            retval12 = tree[:index_parenthesis] + retval1 + "  " + retval2 + " )"
            return retval12
    elif tree[1:7] in [" move1", " move2"]:
        index_parenthesis = tree.index('(', 2)
        arg = getarg(tree[index_parenthesis:])
        retval = remove_silent_heads(arg, algorithm_type, sh_perc)
        if retval == "_sh_":
            return "_sh_"
        else:
            retval = tree[:index_parenthesis] + retval + " )"
            return retval 
    elif tree[2] == '[':
        index_right_bracket = tree.index(']')
        sh = tree[2:index_right_bracket+1]
        if sh_deletion_decision(sh, algorithm_type, sh_perc) == True:
            return "_sh_"
        else: return tree[:index_right_bracket+5]
    else:
        index_end_string = tree.index(')')
        word_string = tree[:index_end_string+1]
        return word_string

def getarg(tree):
    counter = 0
    for index, character in enumerate(tree):
        if character == '(': counter += 1
        if character == ')': 
            counter -= 1
            if counter == 0: 
                return tree[:index+1]
    print("Error: expected )")
    return ""

def sh_deletion_decision(sh, algorithm_type, sh_perc):
    #comment out following line if all silent heads need to be removed:
    if algorithm_type == 0:
        return False
    elif algorithm_type == -1:
        return True
    elif algorithm_type == -3:
        return random.randint(1, 100)<=sh_perc*100
    elif algorithm_type > 0:
        return random.randint(1, 100)<=algorithm_type
    elif algorithm_type == -2:  # algorithm_type == -2
        deletion_list = get_setting("Deletion_Set")
        if sh in deletion_list:
            return True
        else:
            return False
    else:
        raise ValueError(f"no such algorithm type as {algorithm_type}")

def calc_sh_del_perc(tree, deletion_set):
    sh_del_counter = 0
    list_found_shs = re.findall(r"\[[a-z]+\]", tree)
    total_shs = len(list_found_shs)
    for element in list_found_shs:
        if element in deletion_set:
            sh_del_counter += 1
    if total_shs == 0:
        return 0
    else:
        del_perc = (sh_del_counter / total_shs)
        return del_perc


def string2term(string: str, minimalist_algebra, brackets: str = None):
    """
    Wrapper function turning a string representation of a term into a term
    @param minimalist_algebra:
    @param brackets:
    @param prepare:
    @param string: str in the format printed out by nltk.Tree.pformat with
                    parameters margin=100000000000, indent=0, nodesep="",
                                parens="[]", quotes=False
    @return: AlgebraTerm with operations MinimalistFunction using inner algebra
                HMIntervalPairsAlgebra and mover store ListMovers
    """
    if brackets is None:
        brackets = "[]"  # () are in the intervals, so this is safer
    log("string to nltk tree")
    try:
        t = nltk.Tree.fromstring(string, brackets=brackets,
                                 remove_empty_top_bracketing=True)
    except ValueError:
        # see if we can use a quick fix on brackets
        string = resolve_brackets(string, brackets)
        t = nltk.Tree.fromstring(string, brackets=brackets,
                                 remove_empty_top_bracketing=True)
    log(f"nltk.Tree: {t}")
    log("\nconverting to Tree")
    t = nltk2tree(t)
    log(f"Tree: {t}")
    log("\n converting to term")
    t = tree2term(t, minimalist_algebra=minimalist_algebra)
    log(f"term: {t}")

    return t


def resolve_brackets(tree_as_string: str, brackets="()"):
    """
    If there are mismatched brackets, remove or add brackets from the end.
    Note this won't always help, but it's a simple band-aid
    @param tree_as_string: tree as read in from tsv file
    @param brackets: string consisting of the opening and the closing bracket, default '()'
    @return: updated tree as string
    """
    open_bracket = brackets[0]
    close_bracket = brackets[1]
    opens = [token for token in tree_as_string if token == open_bracket]
    closes = [token for token in tree_as_string if token == close_bracket]
    difference = len(opens) - len(closes)
    if difference > 0:
        logging.warning(f"Adding {difference} closing brackets")
        while difference > 0:
            tree_as_string += f" {close_bracket}"
            difference -= 1
    elif difference < 0:
        logging.warning(f"Removing {-difference} closing brackets")
        while difference < 0:
            # tree seems to end with space and \n so look at third-last? sometimes second-last??
            closing_bracket_location = -2
            if tree_as_string[closing_bracket_location] != close_bracket:
                logging.warning(f"string ends with {tree_as_string[closing_bracket_location]}, not {close_bracket}")
                return tree_as_string
            tree_as_string = tree_as_string[:closing_bracket_location] + "\n"  # add the \n back in
            difference += 1
    return tree_as_string


def tsv2sentences_and_terms(corpus_path, minimalist_algebra,
                            brackets: str = None) -> list[tuple[int, str, AlgebraTerm]]:
    """
    Reads in a tsv in the form sentence \t term and returns a list of (sentence, term) pairs.
    @param brackets: brackets for nltk interpretation of tree as a string openingbracketclosingbrack. Default '[]'
    @param minimalist_algebra: inner algebra, default HMIntervalPairsAlgebra
    @param corpus_path: path to the tsv file.
    @return: list of triples (index, str, AlgebraTerm over MinimalistAlgebra with ListMovers and String Triple Algebra).
                If there was an error reading in a line to a term, term is None
    """
    ret = []
    with open(corpus_path, 'r') as f:
        line = f.readline()
        i = 0
        while len(line) > 0:
            current_sentence, term_string = line.split('\t')
            try:
                term = string2term(term_string, minimalist_algebra=minimalist_algebra, brackets=brackets)
            except Exception as e:
                log(f"error creating term from line {i} of {corpus_path}: {e}")
                logging.warning(f"error creating term from line {i} of {corpus_path}: {e}")
                term = None
                raise e
            ret.append((i, current_sentence, term))
            line = f.readline()
            i += 1
    return ret


def text_file2terms(corpus_path, minimalist_algebra, brackets: str = None):
    """
    Reads in a text with one term per line, and returns a list of (index, term) pairs
    @param corpus_path: path to the text file
    @return: list of pairs (index, AlgebraTerm over MinimalistAlgebra with ListMovers and String Triple Algebra)
                term is replaced by error if error was generated
    """
    ret = []
    with open(corpus_path, 'r') as f:
        line = f.readline()
        i = 0
        while len(line) > 0:
            term_string = line
            logging.info(f"line {i}")
            try:
                term = string2term(term_string, minimalist_algebra=minimalist_algebra, brackets=brackets)
            except TypeError as e:
                raise e
            except Exception as e:
                log(f"error creating term from line {i} of {corpus_path}: {type(e).__name__}: {e}")
                logging.debug(f"error creating term from line {i} of {corpus_path}: {type(e).__name__}: {e}")
                term = e
            ret.append((i, term))
            line = f.readline()
            i += 1
    return ret


def term_directory2dict(corpus_path):
    """
    Reads in a directory of printed algebra terms and turns them into a dict of
        terms where the keys are the original (and current) file names minus
        the suffix (e.g. wsj_1526) and the values are dicts from the sentence
        numbers in the original corpus)
    @param corpus_path: str: path to term corpus
    @return: dict of str to str to Minimalist Algebra term over intervals and
                with ListMovers mover stores
    """

    print(f"reading in terms in {corpus_path}...")
    term_dict = defaultdict(lambda: {})

    for file_name in os.listdir(f"{corpus_path}/"):
        print(file_name)
        if file_name.endswith(".terms"):
            file_prefix = file_name.split(".")[0]
            log(f"\nFile: {file_name}")
            with open(f"{corpus_path}/{file_name}") as f:
                data = json.load(f)

            for k in data.keys():
                t = string2term(data[k])
                term_dict[file_prefix][k] = t
    return term_dict

def get_setting(SettingFlag):
    fd_setting = open("settings.txt", "r")
    line = fd_setting.readline()
    while line[:len(SettingFlag)].upper() != SettingFlag.upper():
        line = fd_setting.readline()
    fd_setting.close()
    if line == "":
        print("Setting value not found")
        return ""
    else:
        pos_equal = line.index('=')
        setting_value = line[pos_equal+1:]
        setting_value = setting_value.strip()
        if setting_value[0] == '"':
            return setting_value[1:len(setting_value)-1]
        else:
            return int(setting_value)
    
if __name__ == "__main__":

    # read in MGbank corpus, transforms trees to terms, prints them to file, and
    # reads them back in to check it worked

    if len(sys.argv) > 2:
        errors_ok = True  # True if you want to write all transformable terms to file, even if they can't be interpreted
        # do the whole corpus
        in_path = sys.argv[1]  # path to parent folder of input corpus
        out_path = sys.argv[2]  # path to parent folder of output corpus
        out_name = sys.argv[3]  # name of output file (.tsv will be added)
        # make and print the algebra terms
        directory2strings_file(in_path, f"{out_path}/{out_name}.tsv", errors_ok=errors_ok)
    else:
        # just do one sentence
        file_number = "0113"
        # sentence number within file (just the order, not the actual number in the file I think)
        i = 6  # None  # use None for all
        # seed_or_auto = "Auto"
        sub_corpus = "dev"
        corpus_path = f".data/processed/mg_bank/split/{sub_corpus}"
        in_path = f"{corpus_path}/{file_number[:2]}/wsj_{file_number}.mrg"
        # in_path = "../corpora/MGBank/wsj_MGbankSeed/23/wsj_2376.mrg"

        log(f"\nFile: {in_path}; sentence #{i}")
        # get the term and sentence, and display the sentence in NLTK
        for j, (p, a) in enumerate(read_corpus_file(in_path)):
            if i is None or i == j:
                outcome, sentence = nltk_trees2string_list_term(p, a, file_number, i)
                print(sentence)
                outcome.to_nltk_tree().draw()

        # write to file
        # append_or_write_anew = "w"  # "w"  # w for write anew, a for append
        # out_path = "../../data/processed/seq2seq/experimenting/"
        # os.makedirs(out_path, exist_ok=True)
        # suffix = "all" if i is None else i
        # out_path += f"{sub_corpus}_{file_number}_{suffix}.tsv"
        # with open(out_path, append_or_write_anew) as out_connection:
        #     one_mgbank_entry2tsv(corpus_path,
        #                          file_number[:2],
        #                          f"wsj_{file_number}.mrg",
        #                          out_connection,
        #                          sentence_number=i)
        #
        # print("written to", out_path)

        # print(Tree.nltk_tree2tree(p).latex_forest())
        # print(Tree.nltk_tree2tree(a).latex_forest())
        # print(outcome.latex_forest())
        # print(outcome.annotated_tree().latex_forest())

        # print(outcome)

        # TODO make tests for:
        # successive-cyclic: seed 2028 sentence 0
        #

    # terms = term_directory2dict(out_path)
    #
    # print(terms.values())
    #
    # errors = []

    # # check a specific file
    # file_prefix = "wsj_0022"
    # key = "0"
    # t = terms[file_prefix][key]
    #
    # with open(f"{in_path}/{file_prefix}.terms") as f:
    #     raw_data = json.load(f)
    #
    # out_term = string2term(raw_data[key])
    # print(out_term)

    # check the terms we read in for completeness and errors
    # for prefix in terms:
    #     for key in terms[prefix]:
    #         try:
    #             c = terms[prefix][key].evaluate().is_complete()
    #             if not c:
    #                 print("\n *** not complete ***")
    #         except Exception as err:
    #             errors.append((prefix, key, err))
    #
    # print(f"Errors in interpreting terms read in from {out_path}:")
    # if len(errors) == 0:
    #     print("None")
    # for e in errors:
    #     print(e)
