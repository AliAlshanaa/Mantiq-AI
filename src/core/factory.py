import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_model_choice():
    """
    Prompt the user to select their preferred AI engine before execution.
    """
    print("\n--- 🤖 Mantiq-AI: Select AI Engine ---")
    print("1. Gemini 2.0 Flash (Google) - Best for Long Context")
    print("2. GPT-4o (OpenAI) - Best for Synthesis & Logic")
    print("3. Llama 3.3 70B (Groq) - Best for Speed & Review")
    
    choice = input("Enter your choice (1-3) [Default: 1]: ").strip()
    
    mapping = {
        "1": "gemini",
        "2": "openai",
        "3": "llama"
    }
    return mapping.get(choice, "gemini")

def create_llm(provider, temperature=0.3):
    """
    Factory method to initialize the selected LLM provider.
    """
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=temperature,
            max_retries=10
        )
    elif provider == "openai":
        return ChatOpenAI(
            model="gpt-4o", 
            temperature=temperature
        )
    elif provider == "llama":
        # Using Groq for ultra-fast Llama 3 inference
        return ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=temperature
        )
    return None