from openai import OpenAI, ChatCompletion
import json
import dotenv

dotenv.load_dotenv()

# Customer service categories that could be injected used in both the prompt instruction and response processing
PRIMARY_CATEGORIES_ = ["Billing", "Technical Support", "Account Management", "General Inquiry"]
SECONDARY_CATEGORIES_ = {
    "Billing": ["Unsubscribe or upgrade", "Add a payment method", "Explanation for charge", "Dispute a charge"],
    "Technical Support": ["Troubleshooting", "Device compatibility", "Software updates"],
    "Account Management": ["Password reset", "Update personal information", "Close account",
                           "Account security"],
    "General Inquiry": ["Product information", "Pricing", "Feedback", "Speak to a human", "Unknown"]
}

class CustomerServiceChatClient:
    def __init__(self):
        self.client = OpenAI()

    def send_query(self, query: str) -> dict:
        prompt = (
            f"""You will be provided with customer service queries. Classify each query into a primary category and a 
            secondary category.

            Provide your output in json format with the keys: primary and secondary.

                Primary categories: { PRIMARY_CATEGORIES_ }

                Secondary categories by primary category as the key : { SECONDARY_CATEGORIES_ }
            """
        )
        query_with_prompt = f"{prompt} User: {query}"

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query_with_prompt}],
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return self.process_response(response)

    # this method is likely unnecessary with proper prompt engineering with some fine tuning.
    @staticmethod
    def process_response(response: ChatCompletion) -> dict:
        primary_categories = PRIMARY_CATEGORIES_
        secondary_categories = SECONDARY_CATEGORIES_

        # Check if response has choices and choices are not empty
        if hasattr(response, 'choices') and response.choices:
            system_response = response.choices[0].message.content
            # print(system_response)
            primary_category = None
            secondary_category = None

            for category in primary_categories:
                if category in system_response:
                    primary_category = category
                    break

            if primary_category:
                for sub_category in secondary_categories[primary_category]:
                    if sub_category in system_response:
                        secondary_category = sub_category
                        break

            return {primary_category: secondary_category}

        else:
            return {"error": "No response received from the model"}

# The main function for mocking up an app interaction with a terminal client
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
        print(json.dumps(response, indent=4))


if __name__ == "__main__":
    main()
