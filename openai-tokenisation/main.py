import tiktoken
import os
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

text= "Airtificial intelligence is evovlving inside the world"
text1= "I love how human use artificial intelligence to solve problems"

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MY_API_KEY")
)

#Load tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

#convert text into token ids
tokens= encoding.encode(text)

decoded = encoding.decode(tokens)

print("original text", text)
print("Token ids",tokens)
print("token length", len(tokens))
print("Decoded text",decoded)


#embeddings
print("\n ------ Embeddings------")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[text,text1]
)

# extracting vector embedding
embedding1= response.data[0].embedding
embedding2= response.data[1].embedding

print("Embedding 1 dimensions ", len(embedding1))
print("Embedding 2 dimensions ", len(embedding2))

#displaying first 10 values
print("\n--- first 10 values of embedding 1")
print(embedding1[:10])
