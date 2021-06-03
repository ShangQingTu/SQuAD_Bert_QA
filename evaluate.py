from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from train import get_processed_features
import os
from tqdm import tqdm
import argparse
import json


def get_bleu_score(src, tgt):
    smooth = SmoothingFunction()
    # default setting is BLEU-4 score
    score = sentence_bleu([src], tgt, smoothing_function=smooth.method1)
    return score


def work(prediction_path, processed_dir, bleu_calculate):
    print('Loading data')
    dev_features_path = os.path.join(processed_dir, "dev.pkl")
    dev_examples, dev_features, dev_dataset = get_processed_features(dev_features_path)

    fin = open(prediction_path, "r")
    predictions = json.load(fin)

    results = []
    for dev_example in tqdm(dev_examples, desc='evaluating by bleu'):
        qas_id = dev_example.qas_id
        answers = [ans_dict['text'] for ans_dict in dev_example.answers]
        prediction = predictions[qas_id]

        bleu_scores = [get_bleu_score(prediction, answer) for answer in answers]
        if bleu_calculate == 'mean':
            score = sum(bleu_scores) / len(bleu_scores)
        else:
            score = max(bleu_scores)
        results.append(score)

    mean_res = sum(results) / len(results)
    print("Evaluate in bleu score, for each possible answer, we use {} strategy to calculate".format(bleu_calculate))
    print("Final bleu mean result is %4f" % (mean_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate SQuAD dataset model result by bleu score.")

    # path parameters
    parser.add_argument('--prediction_path', type=str, default='./checkpoints/time_052521/predictions_epoch1.json',
                        help='Where model predict json result is.')
    parser.add_argument('--processed_dir', type=str,
                        default='./dataset/processed',
                        help='Where processed tensor results are.')

    parser.add_argument('--bleu_calculate', type=str,
                        default='mean',
                        choices=['mean', 'max'],
                        help='A question has many possible answers, how to calculate the bleu score.')

    args = parser.parse_args()

    work(args.prediction_path, args.processed_dir, args.bleu_calculate)
