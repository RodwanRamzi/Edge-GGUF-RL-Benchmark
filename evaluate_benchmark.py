import os
import re
import csv
import json
import time
import requests

PROJECT_DIR = "F:\\My AI Algorithm"
TEST_FILE = os.path.join(PROJECT_DIR, "unseen_50_test.json")
RESULTS_CSV = os.path.join(PROJECT_DIR, "ab_accuracy_report.csv")
URL = "http://localhost:11434/api/generate"

def extract_boxed_answer(text):
    matches = re.findall(r'(?:\\boxed\{|\[)\s*(-?\d+(?:\.\d+)?)\s*(?:\}|\])', text)
    return matches[-1].strip() if matches else "NONE"

def evaluate_model(model_name, tag_label):
    if not os.path.exists(TEST_FILE):
        print(f"[ERROR] Missing {TEST_FILE}. Run create_benchmark_data.py first!")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {model_name} ({tag_label})")
    print(f"Unseen Test Set Size: {len(problems)} multi-step problems")
    print(f"=======================================================")

    correct_count = 0
    total_time = 0.0
    rows = []

    for idx, item in enumerate(problems):
        prompt = item["q"]
        expected = str(item["a"]).strip()
        
        start_t = time.time()
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # Low temperature for deterministic benchmark scoring
                "num_ctx": 4096,
                "num_predict": 3072
            }
        }
        try:
            res = requests.post(URL, json=payload).json()
            full_text = res.get("thinking", "") + "\n" + res.get("response", "")
            pred = extract_boxed_answer(full_text)
        except Exception as e:
            pred = f"ERR:{e}"

        elapsed = time.time() - start_t
        total_time += elapsed
        is_correct = 1 if pred == expected else 0
        correct_count += is_correct
        status = "PASS" if is_correct else f"FAIL (Got: {pred} | Expected: {expected})"

        print(f"[{idx+1:02d}/{len(problems)}] {status} | Time: {elapsed:.2f}s")
        rows.append([tag_label, model_name, idx + 1, expected, pred, is_correct, f"{elapsed:.2f}"])

    accuracy = (correct_count / len(problems)) * 100.0
    print(f"\n--- {tag_label} SUMMARY ---")
    print(f"Score: {correct_count}/{len(problems)} ({accuracy:.2f}%)")
    print(f"Total Evaluation Time: {total_time:.2f}s (Avg: {total_time/len(problems):.2f}s/problem)")

    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Stage", "Model", "Problem_ID", "Expected", "Predicted", "Correct", "Time_Sec"])
        writer.writerows(rows)

if __name__ == "__main__":
    print("Select evaluation mode:")
    print("1 = Baseline 'BEFORE' Test (deepseek-r1:1.5b)")
    print("2 = Trained 'AFTER' Test (deepseek-grpo-math)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        evaluate_model("deepseek-r1:1.5b", "BEFORE_BASELINE")
    elif choice == "2":
        evaluate_model("deepseek-grpo-math", "AFTER_GRPO_28BLK")
    else:
        print("Invalid selection.")
