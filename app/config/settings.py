import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
