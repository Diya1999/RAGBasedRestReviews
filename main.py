from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model=OllamaLLM(model="llama3.2")
template= """You are an expert in answering questions abut a pizza restaurant.
Here are some relevant reviews: {reviews}
Here is the questions to answer: {question}
"""
prompt=ChatPromptTemplate.from_template(template)
chain=prompt | model

while True:
    print("\n\n______________________________________________________________")
    question=input("Ask your question: q to quit: ")
    if question.lower() == 'q':
        break
    reviews=retriever.invoke(question)
    #retrive embeds questions, looks in the vector store for relvant reviews using similarty search and return the most relevant reviews
    result=chain.invoke({"reviews":reviews,"question":"What is the best pizza place in town?"})
    print(result)
