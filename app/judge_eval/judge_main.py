from app.temp import analysis_results
from pathlib import Path

# ---- IMPORT YOUR JUDGE PIPELINE ----
from app.judge_eval.dataset import build_judge_dataset
from app.judge_eval.export import write_jsonl

print(analysis_results)

def main():
    print("▶ Building judge dataset from analyze results...")
    print(f"▶ Total analyze responses: {len(analysis_results)}")

    dataset = build_judge_dataset(
        analysis_results=analysis_results,
        threshold_version="v1.0"
    )

    output_path = Path("data/judge_dataset_v1.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(dataset, str(output_path))

    print("✅ Judge dataset generated")
    print(f"📄 Output file: {output_path}")
    print(f"📊 Total judge rows: {len(dataset)}")


if __name__ == "__main__":
    main()