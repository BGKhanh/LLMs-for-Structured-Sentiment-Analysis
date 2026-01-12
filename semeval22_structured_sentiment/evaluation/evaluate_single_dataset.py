import json
from evaluate import convert_opinion_to_tuple, tuple_f1, convert_opinion_to_tuple, calculate_all_metrics, weighted_score, sent_tuples_in_list
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def analyze_errors(gold_dict, pred_dict, output_file="error_analysis.json"):
    """
    Phân tích chi tiết từng mẫu:
    - Tính F1 score cho từng câu (Sentence-level F1)
    - Liệt kê các tuple bị miss (False Negative)
    - Liệt kê các tuple thừa (False Positive)
    """
    analysis_results = []
    
    for sent_id, g_tuples in gold_dict.items():
        p_tuples = pred_dict.get(sent_id, [])
        
        # 1. Tính số True Positive (Weighted) cho câu này
        weighted_tp = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0
        
        # Check Precision (FP & TP)
        # Những tuple dự đoán nào khớp với Gold?
        matched_preds = []
        for p in p_tuples:
            if sent_tuples_in_list(p, g_tuples, keep_polarity=True, mode="all"):
                score = weighted_score(p, g_tuples, mode="all")
                weighted_tp += score
                tp_count += 1
                matched_preds.append((p, score))
            else:
                fp_count += 1 # Dự đoán sai/thừa
        
        # Check Recall (FN)
        # Những tuple Gold nào KHÔNG được dự đoán?
        missed_golds = []
        for g in g_tuples:
            if not sent_tuples_in_list(g, p_tuples, keep_polarity=True, mode="all"):
                fn_count += 1
                missed_golds.append(g)
                
        # Tính F1 cục bộ cho câu này (Optional, chỉ để tham khảo)
        # Lưu ý: F1 tổng không phải là trung bình cộng của F1 từng câu
        prec = weighted_tp / (len(p_tuples) + 1e-16)
        rec = weighted_tp / (len(g_tuples) + 1e-16)
        f1 = 2 * prec * rec / (prec + rec + 1e-16)

        # Chỉ lưu những câu có lỗi (F1 < 1.0)
        if f1 < 0.999: # Dùng < 0.999 thay vì != 1.0 để tránh lỗi float
            analysis_results.append({
                "sent_id": sent_id,
                "score": f1,
                "details": {
                    "precision": prec,
                    "recall": rec,
                    "false_positives": [str(x) for x in p_tuples if x not in [m[0] for m in matched_preds]],
                    "false_negatives": [str(x) for x in missed_golds],
                    "matched_scores": [m[1] for m in matched_preds] # Score overlap của các cặp đúng
                }
            })
            
    # Sort theo score tăng dần (câu sai nặng nhất lên đầu)
    analysis_results.sort(key=lambda x: x["score"])
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved detailed error analysis to {output_file} ({len(analysis_results)} samples)")

def analyze_polarity_confusion(gold_dict, pred_dict, output_img="polarity_cm.png"):
    """
    Vẽ Confusion Matrix cho Polarity, BAO GỒM CẢ CÁC TRƯỜNG HỢP BỎ SÓT (MISSED).
    Labels: Positive, Negative, Neutral, None (Missed)
    """
    y_true = []
    y_pred = []
    # Thêm nhãn "None" để đại diện cho việc không tìm thấy opinion
    labels = ["Positive", "Negative", "Neutral", "None"] 

    for sent_id, g_tuples in gold_dict.items():
        p_tuples = pred_dict.get(sent_id, [])
        
        # Đánh dấu những tuple dự đoán nào đã được dùng để match
        used_preds = set()

        # 1. Duyệt qua Gold để tìm Match hoặc Missed (FN)
        for i, g in enumerate(g_tuples):
            best_match = None
            best_score = 0
            best_pred_idx = -1
            
            holder1, target1, exp1, pol1 = g
            
            for j, p in enumerate(p_tuples):
                if j in used_preds: continue # Một pred chỉ được match 1 lần
                
                holder2, target2, exp2, pol2 = p
                
                # Logic matching: Ưu tiên Target, backup bằng Expression
                target_match = len(target1.intersection(target2)) > 0
                if len(target1) == 0 or list(target1) == ["_"]:
                     if len(exp1.intersection(exp2)) > 0:
                         target_match = True

                if target_match:
                    # Tính score overlap
                    t_ov = len(target1.intersection(target2)) / (len(target1) + 1e-16)
                    if t_ov >= best_score:
                        best_score = t_ov
                        best_match = p
                        best_pred_idx = j

            if best_match:
                y_true.append(pol1)
                y_pred.append(best_match[3])
                used_preds.add(best_pred_idx) # Đánh dấu đã dùng
            else:
                # Trường hợp Gold có nhưng Pred không tìm thấy => Missed
                y_true.append(pol1)
                y_pred.append("None")

        # 2. (Optional) Những Pred còn thừa chưa được match => False Positive
        # Nếu bạn muốn CM thể hiện cả những cái máy đoán thừa (FP), bỏ comment dòng dưới
        for j, p in enumerate(p_tuples):
            if j not in used_preds:
                y_true.append("None")
                y_pred.append(p[3])

    if not y_true:
        print("No matched tuples found for Polarity Analysis.")
        return

    print("\n" + "="*30)
    print("POLARITY CLASSIFICATION REPORT (WITH MISSED)")
    print("="*30)
    # Lưu ý: 'None' class sẽ kéo metrics xuống thấp vì nó tính cả lỗi Detection
    print(classification_report(y_true, y_pred, labels=labels, digits=3))

    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Polarity Confusion Matrix (including Missed opinions)')
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Polarity Confusion Matrix saved to {output_img}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")
    parser.add_argument("--exclude_file", help="path to json file containing sent_ids to exclude", default=None)
    parser.add_argument("--num_exclude", help="ids to exclude", default=None)
    parser.add_argument("--num_samples", type=int, help="number of samples to evaluate (from the beginning)", default=None)
    parser.add_argument("--error_file", help="path to save error analysis json", default="error_analysis.json") 
    parser.add_argument("--cm_file", help="path to save confusion matrix image", default="polarity_cm.png")

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
            sliced_ids = exclude_ids[:args.num_exclude]
            print(sliced_ids)
            exclude_set = {str(x) for x in sliced_ids}
            
        gold = [s for s in gold if s["sent_id"] not in exclude_set]
        preds = [s for s in preds if s["sent_id"] not in exclude_set]
        print(f"Excluded {len(exclude_ids)} IDs. Remaining samples: {len(gold)}")

    if args.num_samples is not None:
        gold = gold[:args.num_samples]
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

    analyze_polarity_confusion(gold, preds, args.cm_file)

    # analyze_errors(gold, preds, args.error_file)

if __name__ == "__main__":
    main()
