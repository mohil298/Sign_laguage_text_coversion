import re
from collections import defaultdict, Counter

class SpellAndSentenceHelper:
    """
    Hunspell/CyHunspell spelling correction with optional pyspellchecker fallback.
    Next-word suggestions are a simple bigram/trigram baseline, not a translator.
    """
    def __init__(self, corpus_path=None):
        self.spell = None
        self.backend = "none"
        try:
            from cyhunspell import Hunspell
            self.spell = Hunspell()
            self.backend = "cyhunspell"
        except Exception:
            try:
                from spellchecker import SpellChecker
                self.spell = SpellChecker()
                self.backend = "pyspellchecker"
            except Exception:
                pass

        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        if corpus_path:
            self.train_ngram(corpus_path)

    def train_ngram(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = re.findall(r"[A-Za-z']+", line.lower())
                for i in range(len(words) - 1):
                    self.bigram[words[i]][words[i+1]] += 1
                for i in range(len(words) - 2):
                    self.trigram[(words[i], words[i+1])][words[i+2]] += 1

    def correct_word(self, word):
        if not self.spell or not word.isalpha():
            return word
        try:
            # CyHunspell API
            if self.backend == "cyhunspell":
                if self.spell.spell(word):
                    return word
                suggestions = self.spell.suggest(word)
                return suggestions[0] if suggestions else word
            # pyspellchecker API
            candidate = self.spell.correction(word)
            return candidate or word
        except Exception:
            return word

    def correct_text(self, text):
        tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
        return "".join(self.correct_word(t) if t.isalpha() else t for t in tokens)

    def next_words(self, text, limit=3):
        words = re.findall(r"[A-Za-z']+", text.lower())
        if not words:
            return []
        counts = self.trigram[(words[-2], words[-1])] if len(words) >= 2 else Counter()
        if not counts:
            counts = self.bigram[words[-1]]
        return [w for w, _ in counts.most_common(limit)]
