from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=1,
    messages=[{"role": "user", "content": "Create a book of sarcastic dad jokes with chapters for cars, space, "
                                          "family, music, and food with 10 jokes per chapter."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")