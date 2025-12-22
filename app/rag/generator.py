from groq import Groq
from app.rag.prompts import SYSTEM_PROMPT
from app.config.settings import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def generate(question: str, context: str):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )
    return response.choices[0].message.content.strip()
