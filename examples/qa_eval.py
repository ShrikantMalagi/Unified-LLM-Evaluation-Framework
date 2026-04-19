from core.model import CallableModel
from runners.evaluate import evaluate


def main():
    dataset = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
    ]

    model = CallableModel(name="demo-model", fn=lambda prompt: "4" if "2+2" in prompt else "Paris")

    result = evaluate(dataset, model, task_type="qa")
    print(result)


if __name__ == "__main__":
    main()
