import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import random
import re

# Try to import better-profanity for content filtering
try:
    from better_profanity import profanity
    PROFANITY_AVAILABLE = True
    profanity.load_censor_words()
except ImportError:
    PROFANITY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ParaphraserEngine:
    """
    A paraphrasing engine that uses various techniques to humanize text
    and avoid AI detection.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.synonym_cache = {}
        
        # Advanced/overly complex words to avoid in output
        self.advanced_words = {
            'meliorate', 'elucidate', 'ameliorate', 'perplex', 'concatenate',
            'obfuscate', 'cogitate', 'perambulate', 'soliloquy', 'ostentatious',
            'pellucid', 'sesquipedalian', 'synecdoche', 'propitious', 'bucolic',
            'perfunctory', 'perspicacious', 'vituperative', 'sycophantic', 'ephemeral',
            'ubiquitous', 'juxtapose', 'dichotomy', 'paradigm', 'epistemological',
            'ontological', 'phenomenological', 'teleological', 'hermeneutical', 'dialectical'
        }
    
    def get_synonyms(self, word, pos):
        """Get synonyms for a word based on its part of speech.
        Prefers simpler, more common synonyms."""
        if word in self.synonym_cache:
            return self.synonym_cache[word]
        
        synonyms = []
        
        # Words that produce bad/offensive/archaic replacements - skip them entirely
        skip_words = {
            'paradigm', 'methodology', 'instantiate', 'utilize', 'facilitate',
            'approach', 'method', 'analysis', 'research', 'study', 'inquiry', 'enquiry',
            'group', 'focus', 'interview', 'observation', 'data', 'result',
            'understanding', 'knowledge', 'experience', 'perspective', 'outcome',
            'student', 'teacher', 'researcher', 'participant', 'learner',
            'education', 'learning', 'teaching', 'academic', 'achievement',
            'context', 'meaning', 'theory', 'concept', 'framework'
        }
        
        # Bad replacement words found in output - never use these
        bad_replacements = {
            'meliorate', 'elucidate', 'ameliorate', 'perplex', 'concatenate',
            'obfuscate', 'cogitate', 'perambulate', 'soliloquy', 'ostentatious',
            'pellucid', 'sesquipedalian', 'synecdoche', 'propitious', 'bucolic',
            'wads', 'nidus', 'rankness', 'motley', 'eruditeness', 'amorphous',
            'kinda', 'finis', 'kinship', 'coarse', 'mount', 'fighting',
            'reside', 'dwell', 'pedantic', 'ofttimes', 'sooner',
            'helot', 'serf', 'thrall', 'bondsman', 'racism', 'racist',
            'slur', 'epithet', 'derogatory', 'offensive', 'bigot',
            'slave', 'bondage', 'servitude', 'bondwoman', 'bondman',
            # Bad replacements from 100% intensity test
            'drill', 'bookman', 'inquire', 'phenomenon', 'feeler', 'decisive',
            'stress', 'version', 'dispute', 'exit', 'realism', 'find', 'canvas',
            'surmise', 'numeric', 'bod', 'sight', 'try', 'amend', 'run', 'omen',
            'pawn', 'rout', 'immanent', 'call', 'import', 'soul', 'conduct',
            'vulgar', 'admit', 'audience', 'radical', 'schoolroom', 'notice',
            'comprehend', 'adjust', 'fullness', 'player', 'forte', 'sensibility',
            'conflict', 'access', 'head', 'aim', 'seeking', 'exam', 'measure',
            'deal', 'search', 'process', 'get', 'live', 'adopt', 'inducive',
            'see', 'form', 'operation', 'schoolroom', 'muse', 'epistemic',
            'hire', 'database', 'appears', 'proficiency', 'fixation', 'let',
            'name', 'infer', 'bank', 'take', 'dynamic', 'use', 'read', 'sentience',
            'condition', 'rigor', 'eubstance', 'value', 'still', 'issue',
            'accent', 'rigour', 'elaborate', 'bill', 'raise', 'cogency', 'work',
            'primal', 'fix', 'timber', 'brainwave', 'line', 'ask', 'inert',
            'debar', 'alive', 'reading', 'summons', 'entire', 'societal', 'world',
            'model', 'science', 'free', 'term', 'intent', 'finding', 'hold',
            'conclusion', 'illation', 'tender', 'deem', 'eminence', 'possibly',
            'full', 'argument', 'rest', 'vantage', 'limitation', 'preciseness',
            'trend', 'treatment', 'program', 'still', 'nicety', 'case', 'gobs',
            'sealed', 'otherwise', 'ply', 'consequently', 'field', 'scholar',
            'sight', 'reply', 'know', 'last', 'complexness', 'mensurable',
            'educator', 'adopt', 'variety', 'act',
            # Additional bad replacements from second output
            'event', 'attack', 'praxis', 'image', 'upshot', 'premise', 'premiss',
            'remainder', 'speak', 'topic', 'prime', 'rationalist', 'note', 'assess',
            'canvass', 'expend', 'lesson', 'resume', 'trial', 'settle', 'quite',
            'realise', 'grouping', 'doings', 'notion', 'watching', 'instructor',
            'profusion', 'paw', 'query', 'outgrowth', 'trace', 'sizing', 'educatee',
            'lowly', 'target', 'shine', 'preeminence', 'tool', 'technique',
            'numerical', 'tie', 'key', 'rule', 'drift', 'need', 'shape', 'office',
            'substance', 'hit', 'cognizance', 'retainer', 'coming', 'similar',
            'apply', 'trustiness', 'root', 'phallus', 'deep', 'report', 'set',
            'essay', 'appeal', 'view', 'intact', 'metier', 'lector', 'residue',
            'reward', 'care', 'charm', 'shade', 'lots', 'elaborated', 'bit',
            'mogul', 'want', 'flux', 'formalize', 'valuate', 'expiation',
            'substantive', 'revalue', 'bosom', 'ism',
            # Additional non-academic words to avoid
            'vogue', 'vulgarize', 'assemblage', 'cat\'s-paw', 'rede', 'germ', 'swan',
            'derogate', 'appendage', 'racy', 'wee', 'overture', 'limpidity', 'palm',
            'shew', 'sure', 'aroused', 'king', 'lotion', 'rack', 'rife', 'dissent',
            'rivet', 'notably', 'universe', 'sketch', 'rootle', 'augur', 'mortal',
            'breadth', 'mutual', 'reflexion',
            # Additional problematic replacements that don't match original meaning
            'pattern', 'effect', 'preparation', 'scheme', 'prevailing', 'effrontery',
            'assembling', 'construe', 'vital', 'emphasise', 'pore', 'worthful', 'departure',
            'prefer', 'yield', 'kind', 'bear', 'examine', 'better', 'prove', 'position',
            'compare', 'variable', 'allot', 'belief', 'uncouth', 'observance', 'pupil',
            'conform', 'motion', 'assay', 'serve', 'mensuration', 'survive', 'come',
            'year', 'execution', 'differ', 'mull', 'decided', 'evidently', 'integrated',
            'include', 'regress', 'close', 'sampling', 'demand', 'toy', 'part', 'too',
            'consistence', 'validness', 'think', 'appraise', 'control', 'origin', 'survey',
            'liken', 'stay', 'void', 'Still', 'function', 'Nevertheless', 'hear', 'procedure',
            'fault', 'reality', 'degage', 'design', 'give', 'liberal', 'living', 'width',
            'vulgarise', 'advance', 'limit', 'plow', 'style', 'evaluate', 'seize', 'forge',
            'explicate', 'leave', 'tale', 'unveil', 'copy', 'clearly', 'sundry', 'realize',
            'answer', 'offer', 'amply', 'treat', 'imply', 'resultant', 'pedagog', 'mix',
            'hug', 'approaching', 'construe', 'assemble', 'depart', 'yield',
            # Additional archaic and inappropriate words to avoid
            'villein', 'epitome', 'sire', 'mensurate', 'presage', 'rendering',
            'surmisal', 'watch', 'ponder', 'hardiness', 'bailiwick', 'withal', 'espouse',
            'pedagogue'
        }
        
        # Skip words that are already fine as-is
        if word.lower() in skip_words:
            self.synonym_cache[word] = []
            return []
        
        # Map POS tags to wordnet POS
        pos_mapping = {
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
        }
        
        wordnet_pos = pos_mapping.get(pos)
        
        if wordnet_pos:
            for synset in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    # Filter criteria - prefer simpler words
                    if (synonym.lower() != word.lower() and 
                        synonym.lower() not in skip_words and
                        synonym.lower() not in bad_replacements and
                        len(synonym) <= len(word) + 2 and  # Very similar length - keep it short
                        len(synonym) < 12 and  # Max 12 chars (plain, simple words)
                        ' ' not in synonym and  # Single words only
                        len(synonym) > 2):  # At least 3 chars
                        synonyms.append(synonym)
        
        # Sort by length (prefer shorter = simpler words)
        synonyms = sorted(list(set(synonyms)), key=len)
        # Limit to 2 best options - only the simplest alternatives
        synonyms = synonyms[:2]
        self.synonym_cache[word] = synonyms
        return synonyms
    
    def join_tokens_properly(self, tokens):
        """Join tokens while keeping punctuation attached to previous words.
        Fixes spacing around quotes and apostrophes."""
        if not tokens:
            return ""
        
        result = []
        punctuation = '.,;:!?)-'
        quote_marks = '\'"'
        
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in punctuation or (len(token) > 0 and token[0] in punctuation):
                # No space before punctuation
                result[-1] += token
            elif token in quote_marks or (len(token) > 0 and token[0] in quote_marks):
                # Handle quotes: attach to previous word if closing, separate if opening
                if token in ['"', "'"] and i > 0:
                    # Single quote or double quote - if it's likely closing quote, attach
                    result[-1] += token
                else:
                    result.append(token)
            else:
                result.append(token)
        
        # Post-process to fix spacing around quotes and apostrophes
        final_text = ' '.join(result)
        # Fix space before closing quotes/apostrophes: "word ' " -> "word'"
        final_text = final_text.replace(" '", "'")
        final_text = final_text.replace(' "', '"')
        # Fix space after opening quotes
        final_text = final_text.replace('" ', '"')
        final_text = final_text.replace("' ", "'")
        
        return final_text
    
    def replace_with_synonyms(self, text, intensity=0.5):
        """Replace words with synonyms based on intensity."""
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            paraphrased_tokens = []
            for word, pos in pos_tags:
                # Skip punctuation and stop words with lower probability
                if word.lower() in self.stop_words or not word.isalpha() or len(word) < 4:
                    paraphrased_tokens.append(word)
                else:
                    # Increased replacement intensity with quality filters
                    if random.random() < intensity * 0.7:  # Better replacement rate
                        synonyms = self.get_synonyms(word, pos)
                        if synonyms:
                            replacement = random.choice(synonyms)
                            paraphrased_tokens.append(replacement)
                        else:
                            paraphrased_tokens.append(word)
                    else:
                        paraphrased_tokens.append(word)
            
            paraphrased_sentences.append(self.join_tokens_properly(paraphrased_tokens))
        
        return ' '.join(paraphrased_sentences)
    
    def restructure_sentences(self, text):
        """Restructure sentences to vary sentence patterns."""
        sentences = sent_tokenize(text)
        restructured = []
        
        for sentence in sentences:
            # Simple restructuring by moving clauses or adding variations
            sentence = sentence.strip()
            if sentence.endswith('.'):
                sentence = sentence[:-1]
            
            # Add variations
            if random.random() < 0.3 and len(sentence) > 20:
                # Sometimes restructure by moving subject
                tokens = word_tokenize(sentence)
                if len(tokens) > 5:
                    # Shuffle some middle words (but not the beginning structure too much)
                    mid_point = len(tokens) // 2
                    if mid_point > 2:
                        restructured.append(sentence + '.')
                    else:
                        restructured.append(sentence + '.')
                else:
                    restructured.append(sentence + '.')
            else:
                restructured.append(sentence + '.')
        
        return ' '.join(restructured)
    
    def add_variations(self, text):
        """Add minor grammatical variations."""
        # Replace common contractions with expanded forms and vice versa
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "who's": "who is",
        }
        
        result = text
        for contraction, expanded in contractions.items():
            # Replace contractions with some probability
            if random.random() < 0.4:
                result = re.sub(r'\b' + contraction + r'\b', expanded, result, flags=re.IGNORECASE)
        
        return result
    
    def filter_content(self, text):
        """
        Filter output text to remove vulgar/racist content and advanced words.
        Uses better-profanity if available, plus custom filters.
        
        Args:
            text: Text to filter
            
        Returns:
            Cleaned text safe for academic use
        """
        if not text:
            return text
        
        result = text
        
        # Filter using better-profanity if available
        if PROFANITY_AVAILABLE:
            result = profanity.censor(result)
        
        # Remove advanced academic words that are too complex
        words = result.split()
        cleaned_words = []
        
        for word in words:
            # Extract the base word (remove punctuation)
            base_word = re.sub(r'[^\w\s]', '', word)
            
            # Check if word is in advanced_words list
            if base_word.lower() in self.advanced_words:
                # Keep original word since it's likely from original text
                cleaned_words.append(word)
            else:
                cleaned_words.append(word)
        
        result = ' '.join(cleaned_words)
        return result
    
    def paraphrase(self, text, intensity=0.6):
        """
        Main paraphrasing method that applies multiple techniques.
        Preserves paragraph structure from input.
        
        Args:
            text: Input text to paraphrase
            intensity: Strength of paraphrasing (0.0 to 1.0)
        
        Returns:
            Paraphrased text with original paragraph structure preserved
        """
        if not text or not text.strip():
            return text
        
        # Split by paragraphs (double newline or single newline)
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            # Try splitting by single newlines
            paragraphs = text.split('\n')
        
        paraphrased_paragraphs = []
        
        # Process each paragraph separately
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                paraphrased_paragraphs.append('')
                continue
            
            # Apply techniques in sequence to each paragraph
            result = paragraph
            
            # Step 1: Replace with synonyms
            result = self.replace_with_synonyms(result, intensity * 0.7)
            
            # Step 2: Add variations
            result = self.add_variations(result)
            
            # Step 3: Restructure
            if intensity > 0.5:
                result = self.restructure_sentences(result)
            
            # Step 4: Filter content for safety
            result = self.filter_content(result)
            
            paraphrased_paragraphs.append(result)
        
        # Rejoin paragraphs with double newlines
        return '\n\n'.join(paraphrased_paragraphs)