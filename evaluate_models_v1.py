import json
import time
import requests
import mlflow
from mlflow import log_metric, log_param, log_text

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS =[
    "llama3.2:3b",
    "qwen2.5:3b",
    "gemma2:2b",
]

PROMPTS =[
    {
        "id": "python-001",
        "prompt": "Write a Python function that checks if a number is prime.",
        "expected": ["def ", "return"],
    },
    {
        "id": "rag-001",
        "prompt": (
            "Explain retrieval-augmented generation (RAG) "
            "in simple terms and give one practical example."
        ),
        "expected": ["retriev", "generat"],
    },
    {
        "id": "reasoning-001",
        "prompt": (
            "A university has 200 students. 15% are absent. "
            "How many students are present? Explain your calculation."
        ),
        "expected": ["170"],
    },
]

mlflow.set_experiment("Ollama Three-Model Evaluation")

def query_ollama(model:str, prompt:str)-> dict:
    #Run prompt against a local ollama model

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout = 300,
        
    )

    response.raise_for_status()

    result=response.json()

    elapsed=time.perf_counter() -start

    return{
        "answer": result.get("response",""),
        "latency_seconds": round(elapsed, 3),
        "prompt_tokens": result.get("prompt_eval_count", 0),
        "completion_tokens": result.get("eval_count", 0),
    }

def evaluate_answer(answer: str, expected:list[str]) -> float:
    #Simple keyword coverage score, not a full quality evaluation

    answer= answer.lower()

    if not expected:
        return 1.0

    matched = sum(
        1 for keyword in expected if keyword.lower() in answer
    )
    return round(matched/len(expected), 3)

def main():
    all_results=[]

    for model in MODELS:
        print(f"\nEvaluating model: {model}")


        with mlflow.start_run(run_name=model):
            log_param("model", model)
            log_param("prompt_count", len(PROMPTS))
            log_param("temperature", 0)

            latencies =[]
            scores =[]
            errors = 0

            for item in PROMPTS:
                print(f" Prompt: {item['id']}")

                try: 
                    result = query_ollama(model, item["prompt"])
                    score = evaluate_answer(
                        result["answer"],item["expected"]
                    )

                    latencies.append(result["latency_seconds"])
                    scores.append(score)

                    record={
                        "model": model,
                        "prompt_id": item["id"],
                        "latency_seconds": result["latency_seconds"],
                        "keyword_score":score,
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "answer": result["answer"],
                        "error": None,

                    }
                    print(f"    Latency:{result['latency_seconds']}s")
                    print(f"    Keyword score: {score}")
                    print(f"    Answer: {result['answer'][:250]}...")

                except Exception as exc:
                    errors +=1
                    record={
                        "model": model,
                        "prompt_id": item["id"],
                        "error": str(exc),
                    }

                    print(f"    ERROR: {exc}")

                all_results.append(record)

            log_metric("Successful_prompts", len(latencies))
            log_metric("failed_prompts", errors)

            if latencies:
                log_metric(
                    "average_latency_seconds",
                    sum(latencies)/ len(latencies),
                    
                )

            if scores:
                log_metric(
                    "average_keyword_score",
                    sum(scores)/len(scores),
                )
            log_text(
                json.dumps(
                    [
                        r for r in all_results
                        if r["model"]==model
                    ],
                    indent=2,
                ),
                "evaluation_results.json",
            )

    with open("evaluation_summary.json", "w",encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)

    print("\n Evaluation complete")
    print("Results saved to evaluation_summary.json")
    print("Start the MLflow dashboard with: mlflow ui")

if __name__=="__main__":
    main()

