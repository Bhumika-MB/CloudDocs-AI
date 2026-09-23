import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")


client = genai.Client(api_key=api_key)


def generate_answer(question, context):
    """
    Generate an answer using only the retrieved document context.
    """

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        store=False
    )

    return interaction.output_text