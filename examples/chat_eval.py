from core.model import CallableModel
from runners.evaluate import evaluate


def main():
    dataset = [
        {
            "messages": [
                {"role": "system", "content": "Answer in one short sentence."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "answer": "Paris.",
        }
    ]

    model = CallableModel(name="demo-chat-model", fn=lambda prompt: "Paris." if "France" in prompt else "Unknown.")

    result = evaluate(dataset, model, task_type="chat")
    print(result)


if __name__ == "__main__":
    main()
