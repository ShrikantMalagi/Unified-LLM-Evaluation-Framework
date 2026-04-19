from core.model import CallableModel
from runners.evaluate import evaluate


def main():
    dataset = [
        {
            "query": "Who wrote The Hobbit?",
            "context": "J.R.R. Tolkien wrote The Hobbit.",
            "answer": "J.R.R. Tolkien",
        }
    ]

    model = CallableModel(name="demo-model", fn=lambda prompt: "J.R.R. Tolkien")

    result = evaluate(dataset, model, task_type="rag")
    print(result)


if __name__ == "__main__":
    main()
