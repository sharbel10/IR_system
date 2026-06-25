import sys
from pathlib import Path
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class QueryRefinementService:
    def __init__(self):
        self.words_corpus = Counter()
        self.word_pairs = defaultdict(Counter)
        self.load_corpus_from_dataset()

    def load_corpus_from_dataset(self):
        try:
            import pandas as pd
            processed_path = BASE_DIR / 'data' / 'processed'
            documents_file = processed_path / 'documents.csv'
            
            if documents_file.exists():
                df = pd.read_csv(documents_file)
                lines = df['text'].fillna("").astype(str).tolist()[:30000]
                
                for line in lines:
                    words = re.findall(r'\w+', line.lower())
                    self.words_corpus.update(words)
                    for i in range(len(words) - 1):
                        self.word_pairs[words[i]][words[i+1]] += 1
                        self.word_pairs[words[i+1]][words[i]] += 1
        except Exception:
            pass

    def _edits1(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _known(self, words):
        return set(w for w in words if w in self.words_corpus)

    def correct_word_with_context(self, word, context_words):
        if word in self.words_corpus:
            return word
            
        candidates = self._known(self._edits1(word)) or self._known(
            e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)
        ) or [word]
        
        if len(candidates) == 1:
            return list(candidates)[0]
            
        best_candidate = None
        max_context_score = -1
        
        for c in candidates:
            context_score = 0
            for ctx in context_words:
                if ctx in self.word_pairs[c]:
                    context_score += self.word_pairs[c][ctx]
            
            if context_score > max_context_score:
                max_context_score = context_score
                best_candidate = c
                
        if max_context_score == 0:
            return max(candidates, key=lambda w: self.words_corpus[w])
            
        return best_candidate

    def suggest_correction(self, raw_query):
        words = raw_query.split()
        clean_words = [w.lower().strip("?!.,-") for w in words]
        
        corrected_words = []
        for i, word in enumerate(words):
            clean_word = clean_words[i]
            if clean_word.isalpha():
                context = [clean_words[j] for j in range(len(clean_words)) if j != i]
                corrected = self.correct_word_with_context(clean_word, context)
                corrected_words.append(corrected if not word[0].isupper() else corrected.capitalize())
            else:
                corrected_words.append(word)
                
        return " ".join(corrected_words)

    def expand_with_synonyms(self, processed_query, max_synonyms=1):
        words = processed_query.split()
        expanded_words = list(words)
        
        for word in words:
            if len(word) < 4:
                continue
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    syn_word = lemma.name().replace('_', ' ').lower()
                    if syn_word != word and syn_word not in synonyms and syn_word.isalpha():
                        synonyms.append(syn_word)
            
            expanded_words.extend(synonyms[:max_synonyms])
            
        return " ".join(list(dict.fromkeys(expanded_words)))