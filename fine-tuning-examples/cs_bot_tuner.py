from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI()


def main():
    file = client.files.create(
        file=open("cs_bot_training_data.jsonl", "rb"),
        purpose="fine-tune"
    )

    client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-3.5-turbo"
    )


if __name__ == "__main__":
    main()
