import json, pathlib
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from retriever.retriever import retrieve_chunks
from orchestration.flow import call_llm

goldens = []
golden_path = pathlib.Path("evals/goldens.jsonl")
if golden_path.exists():
    with golden_path.open() as f:
        for line in f:
            goldens.append(json.loads(line))

dataset = []
for g in goldens:
    ctx = [c["text"] for c in retrieve_chunks(g["question"], top_k=4)]
    dataset.append({"question": g["question"], "answer": g["answer"], "contexts": ctx})

report = evaluate(dataset=dataset, metrics=[faithfulness, context_precision, context_recall])
print(report)  # prints metric summary
