from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "src/keyword_to_query_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def keyword_to_query(keyword):
    inputs = tokenizer(keyword, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
    
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=32, num_return_sequences=1, do_sample=True)
    
    generated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_query

if __name__=="__main__":
    while True:
        keyword = input("Enter a keyword (or 'quit' to exit): ")
        if keyword.lower() == 'quit':
            break
        
        query = keyword_to_query(keyword)
        print(query)

#"What is the ideal ratio of protein, carbohydrates, and fats in a balanced diet? Please suggest a daily meal plan that reflects these ratios and explain the impact of each nutrient on the body."
#{'daily meal plan': 3.090768337249756, 'ideal ratio': 2.204787492752075, 'balanced diet': 2.0392839908599854}