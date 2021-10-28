import argparse
import sys, os
from collections import Counter
import numpy as np

from pathlib import Path

#Please Change the folded name from LINE 106-113
NO_RELATION = 0
LOOP=10



def score(key, prediction, verbose=True, NO_RELATION=NO_RELATION):

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0

        # for relation in sorted(relations):
        #     longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(' ')
            if prec < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(' ')
            if recall < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(' ')
            if f1 < 1.0:
                sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))



for i in range(LOOP):
    print("Generate")
    key=np.load('./semeval30/self_all_label%d.npy'%i)
    prediction=np.load('./semeval30/self_all_prediction%d.npy'%i)
    print(Counter(prediction))
    print(Counter(key))
    score(key,prediction,verbose=True,NO_RELATION=NO_RELATION)
    print("Pseudo")
    key_pseudo = np.load('./semeval30/self_all_pseudo_label%d.npy' % i)
    prediction_pseudo = np.load('./semeval30/self_all_pseudo_prediction%d.npy' % i)
    print(Counter(prediction_pseudo))
    print(Counter(key_pseudo))
    score(key_pseudo, prediction_pseudo, verbose=True, NO_RELATION=NO_RELATION)
