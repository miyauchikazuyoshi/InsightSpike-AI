
import os
import sys

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from insightspike import create_agent
from insightspike.config.models import LLMConfig

def main():
    print("Testing Local LLM (Ollama) Connection...")
    
    # Configure for Ollama
    # Assumes Ollama is running on localhost:11434 with 'mistral' or similar
    # User might need to change model name
    llm_config = LLMConfig(
        provider="ollama",
        model="mistral",  # Common default, modify as needed
        api_base="http://localhost:11434/v1",
        api_key="ollama", # Dummy key required by OpenAI client
        temperature=0.7
    )
    
    try:
        agent = create_agent(llm=llm_config)
        print(f"Agent created with provider: {agent.config.llm.provider}")
        
        # Simple test
        # We catch connection error if Ollama is not running
        response = agent.process_question("Hello, are you running locally?")
        print("\nResponse from LLM:")
        print(response.response)
        
    except Exception as e:
        print(f"\n[Validation Note] Connection failed (Expected if Ollama is not running): {e}")
        print("To fix: Ensure Ollama is running (`ollama serve`) and has the model (`ollama pull mistral`).")

if __name__ == "__main__":
    main()
