from typing import List, Dict, Any
from openai import AsyncOpenAI
import os
from search import async_search_with_divide
from utils import MAX_CONTEXT_LENGTH
import time
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

async def generate_prompt(query: str, k: int = 5) -> str:
    relevant_chunks = await async_search_with_divide(query, k)
    
    context = ""
    total_length = 0
    
    for chunk in relevant_chunks:
        chunk_text = f"[{chunk['text']}]\n"  
        chunk_length = len(chunk_text)
        
        if total_length + chunk_length > MAX_CONTEXT_LENGTH:
            break
            
        context += chunk_text
        total_length += chunk_length
    
    return PROMPT_TEMPLATE.format(context=context.strip(), query=query)

async def generate_answer(prompt: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
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