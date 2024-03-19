import sys
from minimalist_parser.convert_mgbank.mgbank2algebra import nltk_trees2addressed_list_term, addressed_term2action_list

from minimalist_parser.minimalism.mgbank_input_codec import read_corpus_file, CONJ

# from term_output_codec import term2nltk_tree

if len(sys.argv) > 1:
    corpus_path = sys.argv[1]
else:
    corpus_path = "data/raw_data/MGBank/wsj_MGbankSeed/"

# directory2file(path, "corpora/seeds/")

file = "13/wsj_1319.mrg"

tree_pairs = read_corpus_file(f"{corpus_path}/{file}")
# tree_pairs[0][1].draw()
tree_pairs = [tree_pairs[-1]]

print()
for k, (p, a) in enumerate(tree_pairs):
    outcome, s = nltk_trees2addressed_list_term(p, a)
    print(k, outcome)
    print(s)
    p.draw()

    acts = addressed_term2action_list(outcome, s)
    print("\nActions:")
    for j, act in enumerate(acts):
        print(f"{j}. {act}")

    sentence = outcome.evaluate().spellout()
    print(sentence)

#
# keeps = []
# errors = []
#
# for dir in os.listdir(corpus_path):
#     if os.path.isdir(f"{corpus_path}/{dir}"):
#         for f in os.listdir(f"{corpus_path}/{dir}"):
#             if f.endswith(".mrg"):
#                 log(f"\nFile: {f}")
#                 tree_pairs = read_corpus_file(f"{corpus_path}/{dir}/{f}")
#
#                 for i, (p, a) in enumerate(tree_pairs):
#                     try:
#                         outcome, s = nltk_trees2addressed_list_term(p, a, f, i)
#                         acts = addressed_term2action_list(outcome, s)
#                         for act in acts:
#                             if '+keep' in act[0].name:
#                                 keeps.append((f, i, s, acts))
#                                 break
#                     except Exception as e:
#                         errors.append((f, i, e))
#
# for item in keeps:
#     print(item)
#
# print(f"Number of items with keep: {len(keeps)}")
#
# print(f'number of errors: {len(errors)}')
