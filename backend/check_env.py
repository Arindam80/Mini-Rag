import os
from dotenv import load_dotenv

print("--- Starting Environment Variable Check ---")

# This will load the .env file from the current directory
loaded = load_dotenv()

if not loaded:
    print("\n!!! CRITICAL ERROR: Could not find the .env file in this directory. !!!")
    print("Please make sure a file named '.env' exists here.\n")
else:
    print("\n--- .env file was found and loaded successfully. ---\n")

print("--- Checking the values: ---")
cohere_key = os.getenv("COHERE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

print(f"COHERE_API_KEY: {cohere_key}")
print(f"GROQ_API_KEY:   {groq_key}")
print(f"QDRANT_URL:     {qdrant_url}")
print(f"QDRANT_API_KEY: {qdrant_key}")

print("\n--- Analysis ---")
if not all([cohere_key, groq_key, qdrant_url, qdrant_key]):
    print("RESULT: At least one variable is 'None' (missing). Please check your .env file for typos.")
else:
    print("RESULT: All variables appear to be loaded. The issue might be an incorrect key value.")

print("--------------------------------------")