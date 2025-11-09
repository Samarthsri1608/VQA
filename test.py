import os
from dotenv import load_dotenv

env = load_dotenv()
# ------------------------

print(f"GOOGLE_API_KEY is {'set' if os.getenv('GOOGLE_API_KEY') else 'not set'}")
print(f"Port is set to {os.getenv('PORT')}")  