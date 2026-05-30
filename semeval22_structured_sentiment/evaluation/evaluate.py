#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function, division
import sys
import os
import json
from underthesea import word_tokenize

# from nltk.tokenize.simple import SpaceTokenizer

# tk = SpaceTokenizer()

class VietnameseTokenizer:
    def span_tokenize(self, text):
        """
        Tokenizer tiếng Việt dùng underthesea,
        tách riêng từng từ đơn (bỏ merge multiword tokens)
        và trả về offset (start, end) cho từng từ.
        """
        # B1: Lấy chuỗi token có dấu "_" nối multiword
        tokenized_text = word_tokenize(text, format="text")
        # Ví dụ: "Trường Đại_học Quốc_gia Hà_Nội rất đẹp ."

        # B2: Thay "_" -> " " rồi split lại để có token đơn
        tokens = tokenized_text.replace("_", " ").split()

        token_offsets = []
        cursor = 0
        for tok in tokens:
            start = text.find(tok, cursor)
            if start == -1:
                # fallback nếu token bị trùng lặp trong câu
                start = text.find(tok)
            end = start + len(tok)
            token_offsets.append((start, end))
            cursor = end
        return token_offsets

tk = VietnameseTokenizer()

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets =
    [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> (2,3,4)
    """
    token_idxs = []
    #
    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        intoken = False
        for i, (b, e) in enumerate(token_offsets):
            if b == bidx:
                intoken = True
            if intoken:
                token_idxs.append(i)
            if e == eidx:
                intoken = False
    return frozenset(token_idxs)


def convert_opinion_to_tuple(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = list(tk.span_tokenize(text))
    #
    if len(opinions) > 0:
        for opinion in opinions:
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]
            polarity = opinion["Polarity"]
            #
            holder = convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            opinion_tuples.append((holder, target, exp, polarity))
    return opinion_tuples


def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True, mode="all"):
    """
    mode: "all" (SF1), "holder", "target", "expression", "targeted_strict"
    """
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0: holder1 = frozenset(["_"])
    if len(target1) == 0: target1 = frozenset(["_"])
    
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0: holder2 = frozenset(["_"])
        if len(target2) == 0: target2 = frozenset(["_"])
        
        match = False
        # Logic check overlap dựa trên mode
        if mode == "all":
            if (len(holder1.intersection(holder2)) > 0 
                and len(target1.intersection(target2)) > 0 
                and len(exp1.intersection(exp2)) > 0):
                match = True
        elif mode == "holder":
            if len(holder1.intersection(holder2)) > 0: match = True
        elif mode == "target":
            if len(target1.intersection(target2)) > 0: match = True
        elif mode == "expression":
            if len(exp1.intersection(exp2)) > 0: match = True
        elif mode == "targeted_strict":
            # Targeted F1 yêu cầu Exact Match Target 
            if target1 == target2: match = True

        if match:
            if keep_polarity:
                if pol1 == pol2: return True
            else:
                return True
    return False

def weighted_score(sent_tuple1, list_of_sent_tuples, mode="all"):
    best_overlap = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0: holder1 = frozenset(["_"])
    if len(target1) == 0: target1 = frozenset(["_"])
    
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0: holder2 = frozenset(["_"])
        if len(target2) == 0: target2 = frozenset(["_"])
        
        # Tính overlap từng phần
        h_ov = len(holder2.intersection(holder1)) / len(holder1) if len(holder1) > 0 else 0
        t_ov = len(target2.intersection(target1)) / len(target1) if len(target1) > 0 else 0
        e_ov = len(exp2.intersection(exp1)) / len(exp1) if len(exp1) > 0 else 0
        
        current_overlap = 0
        
        # Logic tính điểm overlap tổng dựa trên mode
        if mode == "all":
            # Điều kiện của SF1: Cả 3 phải khớp mới tính điểm
            if (len(holder2.intersection(holder1)) > 0 and 
                len(target2.intersection(target1)) > 0 and 
                len(exp2.intersection(exp1)) > 0):
                current_overlap = (h_ov + t_ov + e_ov) / 3
        elif mode == "holder":
            if len(holder2.intersection(holder1)) > 0: current_overlap = h_ov
        elif mode == "target":
            if len(target2.intersection(target1)) > 0: current_overlap = t_ov
        elif mode == "expression":
            if len(exp2.intersection(exp1)) > 0: current_overlap = e_ov
        # Targeted strict không dùng weighted score (nó là exact match = 1 hoặc 0)

        if current_overlap > best_overlap:
            best_overlap = current_overlap
            
    return best_overlap

def tuple_precision(gold, pred, keep_polarity=True, weighted=True, mode="all"):
    weighted_tp = []
    tp = []
    fp = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity, mode):
                if weighted and mode != "targeted_strict":
                    weighted_tp.append(weighted_score(stuple, gtuples, mode))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 1e-16)

def tuple_recall(gold, pred, keep_polarity=True, weighted=True, mode="all"):
    weighted_tp = []
    tp = []
    fn = []
    assert len(gold) == len(pred)
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity, mode):
                if weighted and mode != "targeted_strict":
                    weighted_tp.append(weighted_score(stuple, ptuples, mode))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 1e-16)

def tuple_f1(gold, preds, keep_polarity=True, weighted=True, mode="all"):
    prec = tuple_precision(gold, preds, keep_polarity, weighted, mode)
    rec = tuple_recall(gold, preds, keep_polarity, weighted, mode)
    return 2 * (prec * rec) / (prec + rec + 1e-16)

# --- HÀM MỚI: TÍNH TẤT CẢ METRIC CÙNG LÚC ---
def calculate_all_metrics(gold, preds):
    metrics = {}
    
    # 1. SF1 (Sentiment Graph F1) - Chuẩn: Weighted + Polarity
    metrics["SF1"] = tuple_f1(gold, preds, keep_polarity=True, weighted=True, mode="all")
    
    # 2. NSF1 (Non-polar SF1) - Chuẩn: Weighted + No Polarity
    metrics["NSF1"] = tuple_f1(gold, preds, keep_polarity=False, weighted=True, mode="all")
    
    # 3. Span F1 (Holder, Target, Exp) - Chuẩn: Weighted + No Polarity 
    metrics["Holder F1"] = tuple_f1(gold, preds, keep_polarity=False, weighted=True, mode="holder")
    metrics["Target F1"] = tuple_f1(gold, preds, keep_polarity=False, weighted=True, mode="target")
    metrics["Exp F1"]    = tuple_f1(gold, preds, keep_polarity=False, weighted=True, mode="expression")
    
    # 4. Targeted F1 - Chuẩn: Exact Target Match + Polarity (Weighted=False) [cite: 114, 174]
    metrics["Targeted F1"] = tuple_f1(gold, preds, keep_polarity=True, weighted=False, mode="targeted_strict")
    
    return metrics

def main():
    """
    Evaluate monolingual structured sentiment results.
    """
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Paths correspond to what Codalab expects
    submit_dir = os.path.join(input_dir, "res/")
    truth_dir = os.path.join(input_dir, "ref/data")

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, "scores.txt")
    output_file = open(output_filename, "w")

    monolingual_datasets = [
        "norec",
        "multibooked_ca",
        "multibooked_eu",
        "opener_en",
        "opener_es",
        "mpqa",
        "darmstadt_unis",
    ]
    crosslingual_datasets = [
        "opener_es",
        "multibooked_ca",
        "multibooked_eu"
        ]

    for subtask, datasets in [("monolingual", monolingual_datasets),
                              ("crosslingual", crosslingual_datasets)]:
        results = []

        print("{}".format(subtask))
        print("#" * 40)

        for dataset in datasets:
            gold_file = os.path.join(truth_dir, subtask, dataset, "test.json")
            submission_answer_file = os.path.join(submit_dir, subtask, dataset,  "predictions.json")

            # read in gold and predicted data, convert to dictionaries
            # where the sent_ids are keys
            with open(gold_file) as infile:
                gold = json.load(infile)
            gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])

            with open(submission_answer_file) as infile:
                preds = json.load(infile)
            preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

            # make sure they have the same keys
            # Todo: make the error message more useful by including the missing values

            g = set(gold.keys())
            p = set(preds.keys())

            assert g.issubset(p), "missing some sentences: {}".format(g.difference(p))
            assert p.issubset(g), "predictions contain sentences that are not in golds: {}".format(p.difference(g))

            f1 = tuple_f1(gold, preds)
            results.append(f1)
            print("SF1 on {0}: {1:.3f}".format(dataset, f1))
            if subtask == "crosslingual":
                crossdataset = "cross_" + dataset
                output_file.write("{0}: {1:.3f}\n".format(crossdataset, f1))
            else:
                output_file.write("{0}: {1:.3f}\n".format(dataset, f1))

        ave_score = sum(results) / len(results)
        print("Average score: {:.3f}".format(ave_score))
        print()

        if subtask == "crosslingual":
            output_file.write("cross_ave_score: {:.3f}\n".format(ave_score))
        else:
            output_file.write("ave_score: {:.3f}\n".format(ave_score))


if __name__ == "__main__":
    main()
