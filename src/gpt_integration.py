from typing import List, Dict, Any
from openai import AsyncOpenAI
import os
from search import async_search_with_divide
from query_processor import divide_query 
from utils import MAX_CONTEXT_LENGTH
import time
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """Given multiple sub-queries extracted from a complex question, provide a comprehensive answer based on the context provided.

Your role:
1. Understand how the sub-queries relate to each other
2. Extract and connect relevant information from the context
3. Provide a unified, coherent answer that addresses all aspects

Guidelines:
- Ensure each part of the complex question is addressed
- Connect related information across different contexts
- Present information in a logical, flowing manner
- Use transitional phrases to connect different aspects
- Make sure the answer is comprehensive yet concise

Context format:
Each context item is marked with its source identifier and relevance score.
[Source: {doc_id}] {text} (Relevance: {score})

Example format:
Query: Explain the relationship between exercise and metabolism, and how does it affect weight loss?
Sub-queries:
1. How does exercise impact metabolism?
2. What is the connection between metabolism and weight loss?
3. What types of exercise are most effective for weight management?

Answer: Exercise significantly impacts metabolism through multiple mechanisms. During physical activity, the body's energy expenditure increases, burning calories for immediate fuel. More importantly, regular exercise builds lean muscle mass, which increases basal metabolic rate (BMR) - the calories burned at rest. This elevated BMR contributes to long-term weight management.

The relationship between metabolism and weight loss is direct but complex. A higher metabolic rate means more calories burned throughout the day, creating a larger caloric deficit necessary for weight loss. However, the body adapts to exercise over time, requiring progressive challenges to maintain effectiveness.

For weight management, both cardio and strength training play crucial roles. High-Intensity Interval Training (HIIT) is particularly effective as it creates an "afterburn effect," increasing metabolism for hours post-exercise. Strength training is equally important as it builds muscle mass, supporting long-term metabolic health.

Now, using the following context items, please answer the user's complex query:
{context}

User's original query: {query}
Sub-queries identified:
{sub_queries}

Answer:"""

async def generate_prompt(query: str, k: int = 5) -> str:
    relevant_chunks = await async_search_with_divide(query, k)
    
    context = ""
    total_length = 0
    sub_queries = await divide_query(query)
    
    formatted_sub_queries = "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_queries))
    
    for chunk in relevant_chunks:
        chunk_text = f"[Source: {chunk['doc_id']}] {chunk['text']} (Relevance: {chunk['relevance_score']:.2f})\n"
        chunk_length = len(chunk_text)
        
        if total_length + chunk_length > MAX_CONTEXT_LENGTH:
            break
            
        context += chunk_text
        total_length += chunk_length
    
    return PROMPT_TEMPLATE.format(
        context=context.strip(),
        query=query,
        sub_queries=formatted_sub_queries
    )

async def generate_answer(prompt: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert assistant specializing in comprehensive analysis and explanation. Your strength lies in connecting information from multiple sources to provide thorough, well-structured answers."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in generating answer: {e}")
        return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

async def get_gpt_answer(query: str) -> str:
    prompt = await generate_prompt(query, k=20)
    answer = await generate_answer(prompt)
    return answer