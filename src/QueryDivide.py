import asyncio
from keywordAlgorithm import extract_keywords
from KeywordToQuery import keyword_to_query
import time

async def process_keyword(keyword):
    return await asyncio.to_thread(keyword_to_query, keyword)

async def process_query(query, topk):
    keyword_extract_start_time = time.time()
    keywords = extract_keywords(query, topk=topk)
    print(keywords)
    print(f"복합 쿼리 -> 키워드 시간: {time.time() - keyword_extract_start_time:.2f} seconds")
    query_make_start_time = time.time()
    tasks = [process_keyword(keyword) for keyword in keywords]
    results = await asyncio.gather(*tasks)
    print(results)
    print(f"키워드 -> 쿼리 시간: {time.time() - keyword_extract_start_time:.2f} seconds")
    return results

async def divide_query(query, topk=5):
    return await process_query(query, topk)

#Can you explain the basics of quantum computing, recommend a good recipe for homemade pizza dough, and tell me about the economic impacts of climate change on agriculture? Also, I'd love to see a simple Python script that calculates prime numbers.