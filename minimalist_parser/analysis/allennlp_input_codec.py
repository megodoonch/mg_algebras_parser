import csv
import json
import sys

from jsonstream import load


def get_one_prediction(path: str, sentence_number: int, prediction_number: int = 0) -> str:
    """
    Given raw_data read in by read-allennlp_predictions, gets the actual tokens of the best prediction
    @param prediction_number: allennlp gave us the top 10 predictions. Default 0 (best)
    @param sentence_number: file contains a list of predictions for sentences. i gets you the ith sentence's prediction
    @param path: path to the predictions json file output by allennlp predict
    @return: string of the predicted tree asked for
    """
    with open(path) as f:
        iterator = load(f)
        print(iterator.__iter__())
        data = list(iterator)
    return data[sentence_number]["predicted_tokens"][prediction_number]


def get_one_prediction_from_plain_file(path: str, sentence_number: int) -> str:

    with open(path) as f:
        i = 0
        while i < sentence_number - 1:
            f.readline()
            i += 1
        return f.readline()

def json2predictions_file(in_path, out_path):
    """
    given the path to a JSON file of predictions, extracts the first predictions and writes them to txt file,
     one tree per line
    @param in_path: path to JSON file
    @param out_path: where to write the predictions as a text file
    @return:
    """
    with open(in_path) as f:
        iterator = load(f)
        data = list(iterator)
    with open(out_path, 'w') as f:
        for entry in data:
            tree = entry["predicted_tokens"][0]
            f.write(f"{tree}\n")

def get_one_gold_item(path: str, sentence_number: int):
    """
    gets a gold sentence and tree from a TSV file of the form
    sent0\ttree0
    sent1\ttree1
    ...
    Note this works regardless of the format of the file -- it just returns a list of the tab-separated items
    @param path: str: the path to the TSV file
    @param sentence_number: int: returns the ith line of the file
    @return: list of the strings from the `sentence_number`th line of the input file
    """
    with open(path) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        return list(tsv_file)[sentence_number]


def print_one_prediction_with_gold(prediction_path: str, gold_path: str, sentence_number: int):
    """
    Print out the sentence, gold tree, and predicted tree for a given sentence index
    @param prediction_path: str: path to prediction JSON file
    @param gold_path: str: path to gold TSV file
    @param sentence_number: which sentence to print
    """
    gold_sentence = get_one_gold_item(gold_path, sentence_number)
    print(f"Sentence {sentence_number}")
    print(gold_sentence[0])
    predicted_tree = get_one_prediction(prediction_path, sentence_number)
    print(f"length of predicted tree: {len(predicted_tree)}")
    print("pred", end=": ")
    for w in predicted_tree:
        print(w, end=" ")
    print()
    print("gold", end=": ")
    print(gold_sentence[1])


def print_one_prediction_with_gold_from_plain_predictions(prediction_path: str, gold_path: str, sentence_number: int):
    """
    Print out the sentence, gold tree, and predicted tree for a given sentence index
    @param prediction_path: str: path to prediction text file
    @param gold_path: str: path to gold TSV file
    @param sentence_number: which sentence to print
    """
    gold_sentence = get_one_gold_item(gold_path, sentence_number)
    print(f"Sentence {sentence_number}")
    print(gold_sentence[0])
    print("predicted")
    print(get_one_prediction_from_plain_file(prediction_path, sentence_number))
    print("gold")
    print(gold_sentence[1])




if __name__ == "__main__":

    # json2predictions_file(sys.argv[1], sys.argv[2])

    if len(sys.argv) >= 4:
        predictions = sys.argv[1]
        gold = sys.argv[2]
        n = int(sys.argv[3])
        try:
            if sys.argv[4]:
                plain = True
        except IndexError:
            plain = False
    else:
        predictions = "../../results/predictions/seq2seq/test/test_predictions.json"
        gold = "../../data/processed/seq2seq/official/test/test.tsv"
        n = 1
        plain = False

    if plain:
        print_one_prediction_with_gold_from_plain_predictions(predictions, gold, n)
    else:
        print_one_prediction_with_gold(predictions, gold, n)
