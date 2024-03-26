from openai import OpenAI
import json
import dotenv

dotenv.load_dotenv()

class CustomerServiceChatClient:
    def __init__(self):
        self.client = OpenAI()

    def send_query(self, query: str) -> str:
        prompt = (
            f"""
            You will be provided with customer service queries. Classify each query into a primary category and a secondary category.
            Provide your output in json format with the primary category as the key and the secondary category as the value.
            Primary categories: Billing, Technical Support, Account Management, General Inquiry
            Secondary categories by primary category as the key:
            Billing: Unsubscribe or upgrade, Add a payment method, Explanation for charge, Dispute a charge,
            Technical Support: Troubleshooting, Device compatibility, Software updates,
            Account Management: Password reset, Update personal information, Close account, Account security,
            General Inquiry: Product information, Pricing, Feedback, Speak to a human, Unknown
            """
        )
        query_with_prompt = f"{prompt} User: {query}"

        response = self.client.chat.completions.create(
            model="[your model id]", # This is the fine-tuned base model id
            messages=[{"role": "user", "content": query_with_prompt}],
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

def main():
    client = CustomerServiceChatClient()

    print("Welcome to the Customer Service Chat Assistant!")
    print("Type 'exit' to quit.")

    while True:
        query = input("Please enter your customer service query: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        response = client.send_query(query)
        print(response)


if __name__ == "__main__":
    main()
