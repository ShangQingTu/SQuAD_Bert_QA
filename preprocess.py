import os
import argparse
import pickle
from transformers.data.processors.squad import SquadV1Processor, squad_convert_examples_to_features
from transformers import AutoTokenizer


def process(args, squadV1Processor, tokenizer, is_training):
    if is_training:
        examples = squadV1Processor.get_train_examples(args.in_dir)
    else:
        examples = squadV1Processor.get_dev_examples(args.in_dir)
    features, dataset = squad_convert_examples_to_features(examples, tokenizer,
                                                           args.max_seq_length,
                                                           args.doc_stride,
                                                           args.max_query_length,
                                                           is_training=is_training,
                                                           return_dataset='pt')
    # save as pkl file
    if is_training:
        features_path = os.path.join(args.out_dir, "train.pkl")
    else:
        features_path = os.path.join(args.out_dir, "dev.pkl")
    with open(features_path, "wb") as fout:
        pickle.dump(examples, fout)
        pickle.dump(features, fout)
        pickle.dump(dataset, fout)

    print("Finish processing one dataset, save features at: ")
    print(features_path)


def work(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    squadV1Processor = SquadV1Processor()
    process(args, squadV1Processor, tokenizer, is_training=False)
    process(args, squadV1Processor, tokenizer, is_training=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Transfer SQuAD data to torch tensor using BertTokenizer.")

    # path parameters
    parser.add_argument('--in_dir', type=str, default='./dataset/',
                        help='Where SQuAD dataset is.')
    parser.add_argument('--out_dir', type=str,
                        default='./dataset/processed',
                        help='Where tensor results goes.')
    parser.add_argument(
        "--cache_dir",
        default="./pretrained_models",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from transformers",
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

    # data parameters
    parser.add_argument(
        "--max_seq_length",
        default=192,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)

    work(args)
