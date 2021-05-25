from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import argparse
import pickle
import time
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import squad_evaluate, compute_predictions_logits
from utils import setup_logger, MetricLogger
from model import BertPasReader


def get_bleu_score(src, tgt):
    smooth = SmoothingFunction()
    # default setting is BLEU-4 score
    score = sentence_bleu([src], tgt, smoothing_function=smooth.method1)
    return score


def get_processed_features(features_path):
    with open(features_path, "rb") as fin:
        examples = pickle.load(fin)
        features = pickle.load(fin)
        dataset = pickle.load(fin)

        return examples, features, dataset


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def validate(dev_loader, dev_features, dev_examples, model, device, save_dir, epoch_num, args):
    # answer store file
    # ans_file_path = os.path.join(save_dir, "epoch{}_valid_answer.txt".format(str(epoch_num)))
    # fout = open(ans_file_path, "w")
    all_results = []

    epoch_iterator = tqdm(dev_loader, desc=" validate - Iteration")

    for batch_iter, valid_batch in enumerate(epoch_iterator):
        model.eval()
        with torch.no_grad():
            input_id, attention_mask, token_type_id, example_indexes, cls_index, p_mask = [a.to(device) for
                                                                                           a in
                                                                                           valid_batch]

            outputs = model(input_id, attention_mask, token_type_id)

            for i, feature_index in enumerate(example_indexes):
                output = [to_list(output[i]) for output in outputs]
                predict_start_logits, predict_end_logits = output
                eval_feature = dev_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = SquadResult(unique_id, predict_start_logits, predict_end_logits)
                all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(save_dir, "predictions_epoch{}.json".format(str(epoch_num)))
    output_nbest_file = os.path.join(save_dir, "nbest_predictions_epoch{}.json".format(str(epoch_num)))
    predictions = compute_predictions_logits(
        dev_examples,
        dev_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file=None,
        verbose_logging=True,
        version_2_with_negative=False,
        null_score_diff_threshold=args.null_score_diff_threshold
    )

    results = squad_evaluate(dev_examples, predictions)
    return results


def train(args, logger, model):
    logger.info('[1] Loading data')
    dev_features_path = os.path.join(args.processed_dir, "dev.pkl")
    dev_examples, dev_features, dev_dataset = get_processed_features(dev_features_path)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False)
    train_features_path = os.path.join(args.processed_dir, "train.pkl")
    train_examples, train_features, train_dataset = get_processed_features(train_features_path)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    logger.info('length of train/valid per gpu: %d/%d' % (len(train_loader), len(dev_loader)))

    logger.info('[2] Building model')
    device = torch.device(args.device)
    model.to(device)
    logger.info(model)
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    meters = MetricLogger(delimiter="  ")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.schedule_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    logger.info('[3] Start training......')

    for epoch_num in range(args.max_epoch):
        model.train()
        epoch_iterator = tqdm(train_loader, desc=" training - Iteration")

        for batch_iter, train_batch in enumerate(epoch_iterator):
            progress = epoch_num + batch_iter / len(train_loader)

            input_id, attention_mask, token_type_id, start_position, end_position, cls_index, p_mask = [a.to(device) for
                                                                                                        a in
                                                                                                        train_batch]
            start_logits, end_logits = model(input_id, attention_mask, token_type_id)

            loss_sum = criterion(start_logits, start_position) + criterion(end_logits, end_position)

            loss = loss_sum / 2.0

            optimizer.zero_grad()

            loss.backward()

            if args.clip_value > 0:
                nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
            if args.clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters.update(loss=loss)

            score = validate(dev_loader, dev_features, dev_examples, model, device, args.save_dir, epoch_num, args)

            if (batch_iter + 1) % (len(train_loader) // 100) == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "progress: {prog:.2f}",
                            "{meters}",
                        ]
                    ).format(
                        prog=progress,
                        meters=str(meters),
                    )
                )

        score = validate(dev_loader, dev_features, dev_examples, model, device, args.save_dir, epoch_num, args)
        logger.info("val")
        logger.info(score)
        save = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step()

        torch.save(save,
                   os.path.join(args.save_dir, 'model_epoch%d_val%.3f.pt' % (epoch_num, score['exact'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train QA model on SQuAD dataset.")

    # path parameters
    parser.add_argument('--squad_dir', type=str, default='./dataset/',
                        help='Where SQuAD dataset is.')
    parser.add_argument('--processed_dir', type=str,
                        default='./dataset/processed',
                        help='Where processed tensor results are.')
    parser.add_argument(
        "--cache_dir",
        default="./pretrained_models",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from transformers",
    )
    parser.add_argument(
        "--save_dir",
        default="./checkpoints/",
        type=str,
        help="Where do you want to store the trained checkpoints and logs",
    )
    # model parameters
    parser.add_argument(
        "--config_name",
        default='albert-base-v2',
        # default='prajjwal1/bert-tiny',
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )

    parser.add_argument(
        "--model_name_or_path",
        default='albert-base-v2',
        # default='prajjwal1/bert-tiny',
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    # tokenizer parameters
    parser.add_argument(
        "--tokenizer_name",
        default='albert-base-v2',
        # default='prajjwal1/bert-tiny',
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_false", help="Set this flag if you are using an uncased model."
    )

    # train parameters
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument("--train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dev_batch_size", default=1, type=int, help="Batch size per GPU/CPU for eval.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device of torch.')
    parser.add_argument('--lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('--schedule_step', type=int, nargs='+', default=[1])
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay rate per batch')
    parser.add_argument('--seed', type=int, default=666666, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optim', default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--clip_norm', type=float, default=2.0)
    parser.add_argument('--use_rl', action='store_true')
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    args.save_dir = os.path.join(args.save_dir, "time_" + time.strftime("%m%d%H"))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    logger = setup_logger("SQuAD-QA", args.save_dir)
    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    # bert config
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    basemodel = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model = BertPasReader(basemodel, config)

    train(args, logger, model)
