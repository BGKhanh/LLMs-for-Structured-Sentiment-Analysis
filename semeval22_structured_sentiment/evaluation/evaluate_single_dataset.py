import json
from evaluate import convert_opinion_to_tuple, tuple_f1, convert_opinion_to_tuple, calculate_all_metrics
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")
    parser.add_argument("--exclude_file", help="path to json file containing sent_ids to exclude", default=None)
    parser.add_argument("--num_samples", type=int, help="number of samples to evaluate (from the beginning)", default=None)

    args = parser.parse_args()

    with open(args.gold_file, encoding="utf-8") as o:
        gold = json.load(o)

    with open(args.pred_file, encoding="utf-8") as o:
        preds = json.load(o)

    for s in gold:
        s["sent_id"] = str(s["sent_id"])
    for s in preds:
        s["sent_id"] = str(s["sent_id"])
        
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8") as f:
            exclude_ids = json.load(f)
            sliced_ids = exclude_ids[:17]
            print(sliced_ids)
            # Tạo set chứa cả int và string để đảm bảo khớp ID bất kể định dạng
            exclude_set = {str(x) for x in sliced_ids}
            
        gold = [s for s in gold if s["sent_id"] not in exclude_set]
        preds = [s for s in preds if s["sent_id"] not in exclude_set]
        print(f"Excluded {len(exclude_ids)} IDs. Remaining samples: {len(gold)}")

    if args.num_samples is not None:
        gold = gold[:args.num_samples]
        # Lọc preds để chỉ giữ lại các sent_id có trong danh sách gold đã cắt
        # Điều này đảm bảo assert p.issubset(g) không bị lỗi
        gold_ids = set(s["sent_id"] for s in gold)
        preds = [s for s in preds if s["sent_id"] in gold_ids]
        print(f"Limited to first {len(gold)} samples.")

    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])

    preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    g = set(gold.keys())
    p = set(preds.keys())

    assert g.issubset(p), f"missing some sentences: {g.difference(p)}"
    assert p.issubset(g), f"predictions contain sentences that are not in golds: {p.difference(g)}"

    # f1 = tuple_f1(gold, preds)
    # print("Sentiment Tuple F1: {0:.3f}".format(f1))
    results = calculate_all_metrics(gold, preds)
    
    print("-" * 30)
    print(f"{'METRIC':<15} | {'SCORE':<10}")
    print("-" * 30)
    for name, score in results.items():
        print(f"{name}: {score:.3f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
