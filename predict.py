# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Tuple, Pad
import paddle.fluid as fluid
from sklearn.metrics import f1_score
from utils import convert_example
from time import time
import pandas as pd


# yapf: disable
parser = argparse.ArgumentParser()
# python predict.py --params_path './checkpoint/model_5600/model_state.pdparams'
parser.add_argument("--params_path", type=str, required=False, default="./checkpoint/model_2500/model_state.pdparams", help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", type=int, default=64, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, tokenizer, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        example = {"text": text}
        input_ids, token_type_ids = convert_example(
            example,
            tokenizer,
            max_seq_length=args.max_seq_length,
            is_test=True)
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        # labels = [label_map[i] for i in idx]
        results.extend(idx)
    return results


def predict_onepiece(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    for text in data:
        example = {"text": text}
        input_ids, token_type_ids = convert_example(
            example,
            tokenizer,
            max_seq_length=args.max_seq_length,
            is_test=True)
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]
    # print('batches: ', batches)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        # print(input_ids, token_type_ids)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        # print('logits: ', logits)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        # print(probs, idx)
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    #test_data = pd.read_csv('./dataset/test_withnoise.csv', sep=',')
    test_data = pd.read_csv('./dataset/dev_withlabel.csv', sep=',')
    #test_data = test_data.loc[test_data.is_noise == 0]
    data = test_data.text.to_list()

    model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
        'ernie-1.0', num_classes=25)
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    # if args.params_path and os.path.isfile(args.params_path):
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("********* Loaded parameters from %s *********" % args.params_path)

    model.eval()
    results = predict(model, data, tokenizer)
    print(results[:5], len(results))
    test_data['label_pre'] = results
    f1 = f1_score(test_data['label'],test_data['label_pre'],average='macro')
    print('macro f1:',f1)
    # print(len(test_data.loc[test_data.label == test_data.label]))\
    #test_data.to_csv('./dataset/pre.csv')
