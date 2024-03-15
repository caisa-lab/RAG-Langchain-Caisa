import argparse
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
from creds import creds
import os

together_api_key = creds['TOGETHER_API_KEY']   
os.environ['OPENAI_API_KEY'] = creds['api_key']


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("********* Prompt *********")
    print(prompt)
    print("********* End Prompt *********")
    
    # Store keywords that will be passed to the API
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"}
    print("header:  ", headers)
    
    # Choose the model to call
    model="togethercomputer/llama-2-7b-chat"

    # Add instruction tags to the prompt
    prompt = f"[INST]{prompt}[/INST]"

    # Set temperature and max_tokens
    temperature = 0.0
    max_tokens = 1024
    url = "https://api.together.xyz/inference"

    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    import requests
    response = requests.post(url,
                         headers=headers,
                         json=data)
    print(response.json()['output'])
    
if __name__ == "__main__":
    main()
