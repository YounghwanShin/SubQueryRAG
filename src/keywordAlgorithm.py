"""
전제: NLTK 'punkt', 'averaged_perceptron_tagger', 'wordnet', stopwords 설치
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')
+) 키워드 추출 알고리즘 관련 패키지 추가 설치
# python -m spacy download en_core_web_sm
# pip install rake-nltk
# pip install git+https://github.com/LIAAD/yake
"""

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
from collections import defaultdict, Counter
import math
import re
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import spacy
from rake_nltk import Rake
import yake


MAX_ITERATIONS = 50
WINDOW_SIZE = 4
DAMPING_FACTOR = 0.85
CONVERGENCE_THRESHOLD = 0.0001

def clean_text(text):
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"n't", " not", text)
    return ''.join(char.lower() for char in text if char in string.printable)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  
    elif treebank_tag.startswith('V'):
        return 'v'  
    elif treebank_tag.startswith('N'):
        return 'n' 
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

def lemmatize_text(text_with_pos, lemmatizer=WordNetLemmatizer()):
    return [lemmatizer.lemmatize(word, pos=pos) if pos else word 
            for word, pos in ((word, get_wordnet_pos(tag)) for word, tag in text_with_pos)]

def prepare_stopwords():
    return set(stopwords.words('english')).union(string.punctuation)

def remove_stopwords(lemmatized_text, stopwords):
    return [word for word in lemmatized_text if word not in stopwords and len(word) > 1]

def build_graph(vocabulary, processed_text):
    vocab_len = len(vocabulary)
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    weighted_edge = defaultdict(Counter)

    for i in range(len(processed_text) - WINDOW_SIZE + 1):
        window = processed_text[i:i + WINDOW_SIZE]
        for j, word1 in enumerate(window):
            for k, word2 in enumerate(window[j+1:], start=j+1):
                if word1 != word2:
                    weight = 1 / (k - j)
                    weighted_edge[word_to_index[word1]][word_to_index[word2]] += weight
                    weighted_edge[word_to_index[word2]][word_to_index[word1]] += weight

    return vocab_len, np.ones(vocab_len, dtype=np.float32), weighted_edge

def calculate_vertex_score(weighted_edge):
    return {node: sum(edges.values()) for node, edges in weighted_edge.items()}

def update_scores(score, vocab_len, weighted_edge, vertex_score):
    for _ in range(MAX_ITERATIONS):
        prev_score = score.copy()
        for i in range(vocab_len):
            score[i] = (1 - DAMPING_FACTOR) + DAMPING_FACTOR * sum(
                weighted_edge[j][i] / vertex_score[j] * score[j]
                for j in weighted_edge if i in weighted_edge[j]
            )
        if np.sum(np.abs(prev_score - score)) <= CONVERGENCE_THRESHOLD:
            break
    return score

def partition_phrases(lemmatized_text_with_pos, stopwords):
    phrases = []
    current_phrase = []
    for word, pos in lemmatized_text_with_pos:
        if word in stopwords:
            if current_phrase:
                phrases.append(tuple(current_phrase))
                current_phrase = []
        else:
            if pos in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                if current_phrase and current_phrase[-1][1].startswith('N'):
                    phrases.append(tuple(current_phrase))
                    current_phrase = []
            elif pos.startswith('N') and current_phrase and not current_phrase[-1][1].startswith('N'):
                phrases.append(tuple(current_phrase))
                current_phrase = []
            current_phrase.append((word, pos))
    if current_phrase:
        phrases.append(tuple(current_phrase))
    return phrases

def score_keyphrases(phrases, score, vocabulary, processed_text, tfidf_scores):
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    word_freq = Counter(processed_text)
    phrase_scores = []
    for phrase in phrases:
        if len(phrase) == 1 and phrase[0][1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
            continue
        phrase_score = sum(
            score[word_to_index[word]] * 
            math.log(len(processed_text) / word_freq[word]) * 
            tfidf_scores.get(word, 1) 
            for word, _ in phrase if word in word_to_index
        )
        phrase_scores.append((phrase_score, ' '.join(word for word, _ in phrase)))
    return sorted(phrase_scores, reverse=True)

def extract_keywords(query, algorithm="base", topk=5):
    '''
    키워드 추출 알고리즘 종류
    - "base": 기준이 되는 알고리즘 (ours)
    - "spaCy"
    - "rake"
    - "yake"
    - "nltk"
    '''
    if algorithm == "base":
        cleaned_text = clean_text(query)
        tokenized_text = word_tokenize(cleaned_text)
        text_with_pos = pos_tag(tokenized_text)
        
        lemmatized_text_with_pos = [(lemma, pos) for lemma, pos in zip(lemmatize_text(text_with_pos), [pos for _, pos in text_with_pos])]
        stopwords = prepare_stopwords()
        processed_text = remove_stopwords([word for word, _ in lemmatized_text_with_pos], stopwords)
        
        if not processed_text:
            word_freq = Counter(tokenized_text)
            return [word for word, _ in word_freq.most_common(topk)]
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([" ".join(processed_text)])
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        
        vocabulary = list(set(processed_text))
        vocab_len, score, weighted_edge = build_graph(vocabulary, processed_text)
        
        vertex_score = calculate_vertex_score(weighted_edge)
        score = update_scores(score, vocab_len, weighted_edge, vertex_score)
        
        phrases = set(partition_phrases(lemmatized_text_with_pos, stopwords))
        phrase_scores = score_keyphrases(phrases, score, vocabulary, processed_text, tfidf_scores)
        
        final_keywords = []
        for _, phrase in phrase_scores:
            if phrase not in final_keywords:
                final_keywords.append(phrase)
            if len(final_keywords) == topk:
                break

        if len(final_keywords) < topk:
            word_freq = Counter(tokenized_text)
            additional_words = [word for word, _ in word_freq.most_common(topk) if word not in final_keywords]
            final_keywords.extend(additional_words[:topk - len(final_keywords)])
            
        return final_keywords[:topk]
            
    if algorithm == "spaCy":
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        keywords = [chunk.text for chunk in doc.noun_chunks]
        return keywords[:topk]
    
    if algorithm == "rake":
        rake = Rake()
        rake.extract_keywords_from_text(query)
        keywords = rake.get_ranked_phrases()
        return keywords[:topk]
    
    if algorithm == "yake":
        kw_extractor = yake.KeywordExtractor()
        keywords = kw_extractor.extract_keywords(query)
        return [kw[0] for kw in keywords[:topk]]
    
    if algorithm == "nltk":
        cleaned_text = clean_text(query)
        tokenized_text = word_tokenize(cleaned_text)
        text_with_pos = pos_tag(tokenized_text)

        lemmatized_text_with_pos = [(lemma, pos) for lemma, pos in zip(lemmatize_text(text_with_pos), [pos for _, pos in text_with_pos])]
        stop_words = prepare_stopwords()
        processed_text = remove_stopwords([word for word, _ in lemmatized_text_with_pos], stop_words)

        word_freq = Counter(processed_text)
        keywords = [word for word, _ in word_freq.most_common(topk)]
        return keywords

if __name__ == "__main__":
    query = "Which company among Google, Apple, and Nvidia reported the largest profit margins in their third-quarter reports for 2023"
    algorithm = "base"
    topk_keywords = extract_keywords(query, algorithm, topk=7)
    print(f"Query: {query}")
    print(f"Keywords: {', '.join(topk_keywords)}")
    
"""
What is the ideal ratio of protein, carbohydrates, and fats in a balanced diet? Please suggest a daily meal plan that reflects these ratios and explain the impact of each nutrient on the body.

How does vitamin C affect iron absorption? Based on this interaction, suggest food combinations that optimize iron intake.

What are the differences between soluble and insoluble dietary fiber, and what are their respective health benefits? Please provide examples of meals that include both types of fiber.

What are the health impacts of specific nutrient deficiencies (e.g., vitamin D, omega-3 fatty acids)? What dietary recommendations can help prevent these deficiencies?

Can you explain the basics of quantum computing, recommend a good recipe for homemade pizza dough, and tell me about the economic impacts of climate change on agriculture? Also, I'd love to see a simple Python script that calculates prime numbers.
"""


#spaCy
#nltk
#Rake
#Yake
#실험