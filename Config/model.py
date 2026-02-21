from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_model():
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

def get_embeddings():
    return MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
