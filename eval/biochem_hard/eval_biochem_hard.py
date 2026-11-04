"""
BioChem-Hard Benchmark Suite
Comprehensive, high-difficulty evaluation for Biology and Chemistry reasoning.
Draws from MMLU-Pro (8-10 options per question) and College-level domain exams (College Biology, College Chemistry, Medical Genetics).
"""
import os
import sys
import json
import time
import random
import re
import argparse
import urllib.request
import concurrent.futures as cf

os.environ["HF_HOME"] = r"G:\hf_cache"

LETTERS = "ABCDEFGHIJ"
ANS_RE = re.compile(r"answer is \(?([A-J])\)?", re.I)

def format_question_mmlu_pro(row):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    prompt = (
        f"{row['question']}\n\n{opts}\n\n"
        "Finish with \"The answer is (X)\" where X is the letter of the correct option. Be brief."
    )
    return {
        "id": f"mmlu_pro_{row.get('question_id', row.get('id', random.randint(1000, 9999)))}",
        "category": "biology" if row["category"] == "biology" else "chemistry",
        "benchmark": "MMLU-Pro",
        "subdomain": row["category"],
        "prompt": prompt,
        "answer": row["answer"].strip().upper()
    }

def format_question_mmlu(row, subject):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["choices"]))
    prompt = (
        f"{row['question']}\n\n{opts}\n\n"
        "Finish with \"The answer is (X)\" where X is the letter of the correct option. Be brief."
    )
    answer_idx = int(row["answer"])
    answer_letter = LETTERS[answer_idx]
    
    category = "chemistry" if "chem" in subject else "biology"
    return {
        "id": f"mmlu_{subject}_{random.randint(10000, 99999)}",
        "category": category,
        "benchmark": "College Science (MMLU)",
        "subdomain": subject,
        "prompt": prompt,
        "answer": answer_letter
    }

def load_all_questions(n_mmlu_pro_per_cat=250, seed=42):
    from datasets import load_dataset
    
    print("[Loader] Loading MMLU-Pro dataset...", flush=True)
    mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    bio_pro = [r for r in mmlu_pro if r["category"] == "biology"]
    chem_pro = [r for r in mmlu_pro if r["category"] == "chemistry"]
    
    rng = random.Random(seed)
    rng.shuffle(bio_pro)
    rng.shuffle(chem_pro)
    
    selected_bio_pro = bio_pro[:n_mmlu_pro_per_cat]
    selected_chem_pro = chem_pro[:n_mmlu_pro_per_cat]
    
    questions = []
    for r in selected_bio_pro:
        questions.append(format_question_mmlu_pro(r))
    for r in selected_chem_pro:
        questions.append(format_question_mmlu_pro(r))
        
    print(f"[Loader] Loaded {len(selected_bio_pro)} MMLU-Pro Bio and {len(selected_chem_pro)} MMLU-Pro Chem questions.", flush=True)
    
    college_subjects = ["college_biology", "college_chemistry", "medical_genetics"]
    for subj in college_subjects:
        print(f"[Loader] Loading cais/mmlu '{subj}'...", flush=True)
        ds = load_dataset("cais/mmlu", subj, split="test")
        print(f"[Loader] Loaded {len(ds)} questions from '{subj}'.", flush=True)
        for r in ds:
            questions.append(format_question_mmlu(r, subj))
            
    print(f"[Loader] Total BioChem-Hard test bank: {len(questions)} questions.", flush=True)
    return questions

def ask(base_url, q, max_tokens=128):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": q["prompt"]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
            text = data["choices"][0]["message"].get("content") or ""
    except Exception as e:
        return {
            "id": q["id"],
            "category": q["category"],
            "subdomain": q["subdomain"],
            "answer": q["answer"],
            "predicted": None,
            "correct": False,
            "error": str(e)
        }
        
    m = ANS_RE.findall(text) or re.findall(r"\b([A-J])\b\W*$", text.strip())
    pred = m[-1].upper() if m else None
    
    return {
        "id": q["id"],
        "category": q["category"],
        "subdomain": q["subdomain"],
        "answer": q["answer"],
        "predicted": pred,
        "correct": (pred == q["answer"]),
        "raw_response": text.strip()[:200]
    }

def run_evaluation(base_url, questions, workers=8):
    print(f"\n[Evaluator] Running BioChem-Hard evaluation across {len(questions)} questions with {workers} workers...", flush=True)
    t0 = time.time()
    
    results = []
    completed = 0
    total = len(questions)
    
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futs = {executor.submit(ask, base_url, q): q for q in questions}
        for f in cf.as_completed(futs):
            res = f.result()
            results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - t0
                qps = completed / max(elapsed, 0.1)
                corr = sum(1 for r in results if r["correct"])
                acc = (corr / completed) * 100
                print(f"[{completed:3d}/{total:3d}] Acc: {acc:5.1f}% ({corr}/{completed}) | {qps:4.1f} q/s | {elapsed:5.1f}s elapsed", flush=True)
                
    elapsed_total = time.time() - t0
    
    # Compute breakdown
    by_subdomain = {}
    by_category = {"biology": {"correct": 0, "total": 0}, "chemistry": {"correct": 0, "total": 0}}
    by_benchmark = {}
    
    for r in results:
        sub = r["subdomain"]
        cat = r["category"]
        if sub not in by_subdomain:
            by_subdomain[sub] = {"correct": 0, "total": 0}
        by_subdomain[sub]["total"] += 1
        if r["correct"]:
            by_subdomain[sub]["correct"] += 1
            
        by_category[cat]["total"] += 1
        if r["correct"]:
            by_category[cat]["correct"] += 1
            
    subdomain_scores = {
        s: {
            "correct": d["correct"],
            "total": d["total"],
            "accuracy": round(100.0 * d["correct"] / max(d["total"], 1), 2)
        }
        for s, d in by_subdomain.items()
    }
    
    category_scores = {
        c: {
            "correct": d["correct"],
            "total": d["total"],
            "accuracy": round(100.0 * d["correct"] / max(d["total"], 1), 2)
        }
        for c, d in by_category.items()
    }
    
    total_correct = sum(1 for r in results if r["correct"])
    overall_acc = round(100.0 * total_correct / max(len(results), 1), 2)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": len(results),
        "total_correct": total_correct,
        "overall_accuracy": overall_acc,
        "elapsed_seconds": round(elapsed_total, 2),
        "category_scores": category_scores,
        "subdomain_scores": subdomain_scores,
        "detailed_results": results
    }
    
    return report

def main():
    parser = argparse.ArgumentParser(description="BioChem-Hard Benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--out", default="biochem_hard_report.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-pro", type=int, default=250, help="Number of MMLU-Pro questions per category")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    questions = load_all_questions(n_mmlu_pro_per_cat=args.n_pro, seed=args.seed)
    report = run_evaluation(args.base_url, questions, workers=args.workers)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n=======================================================", flush=True)
    print(f"  BIOCHEM-HARD EVALUATION COMPLETE", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Overall Accuracy:  {report['overall_accuracy']}% ({report['total_correct']}/{report['total_questions']})")
    print(f"Biology Overall:   {report['category_scores']['biology']['accuracy']}% ({report['category_scores']['biology']['correct']}/{report['category_scores']['biology']['total']})")
    print(f"Chemistry Overall: {report['category_scores']['chemistry']['accuracy']}% ({report['category_scores']['chemistry']['correct']}/{report['category_scores']['chemistry']['total']})")
    print(f"\nSubdomain Breakdown:")
    for sub, d in report['subdomain_scores'].items():
        print(f"  - {sub:20s}: {d['accuracy']:5.1f}% ({d['correct']}/{d['total']})")
    print(f"Saved full report to: {args.out}\n", flush=True)

if __name__ == "__main__":
    main()
