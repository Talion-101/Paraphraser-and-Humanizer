"""
AI Detection Avoidance Module

This module implements techniques to make paraphrased text appear more human-written
and less likely to be flagged by AI content detectors.
"""

import random
import re
from nltk.tokenize import sent_tokenize


class AIDetectionAvoider:
    """
    Techniques to make text less detectable as AI-generated.
    """
    
    def __init__(self):
        # Common human writing patterns (without first-person)
        self.filler_words = [
            "actually", "basically", "kind of", "sort of",
            "arguably", "perhaps", "possibly", "certainly",
            "evidently", "notably", "clearly", "obviously"
        ]
        
        self.transition_phrases = [
            "Furthermore,", "Moreover,", "However,", "On the other hand,",
            "In addition,", "Similarly,", "In contrast,", "As a result,",
            "Consequently,", "Meanwhile,", "Nevertheless,", "Still,"
        ]
        
        self.common_typos = {
            "th": "th",  # No typos - just for structure
        }
    
    def add_human_variations(self, text, intensity=0.5):
        """
        Add human-like variations such as:
        - Occasional informal language
        - Varied sentence structure
        - Natural filler words
        - Imperfect but readable text
        - Preserves paragraph structure
        """
        # Split by paragraphs (empty lines) to preserve structure
        paragraphs = text.split('\n\n')
        modified_paragraphs = []
        
        for para in paragraphs:
            sentences = sent_tokenize(para)
            modified_sentences = []
            
            for i, sentence in enumerate(sentences):
                modified = sentence
                
                # Occasionally add filler words (reduced frequency)
                if random.random() < intensity * 0.1:
                    filler = random.choice(self.filler_words)
                    # Insert filler word at beginning or after first few words
                    if len(modified.split()) > 5 and random.random() < 0.3:
                        words = modified.split()
                        insert_pos = random.randint(1, min(2, len(words)-1))
                        words.insert(insert_pos, filler)
                        modified = " ".join(words)
                    elif modified and modified[0].isupper() and random.random() < 0.2:
                        modified = f"{filler} {modified[0].lower()}{modified[1:]}"
                
                # Occasionally use transition phrases (reduced)
                if i > 0 and random.random() < intensity * 0.08:
                    transition = random.choice(self.transition_phrases)
                    if not modified.startswith(transition):
                        transition_clean = transition.rstrip(',')
                        if modified:
                            modified = f"{transition_clean} {modified[0].lower()}{modified[1:] if len(modified) > 1 else ''}"
                
                # Add occasional parenthetical remarks (less frequent)
                if len(modified.split()) > 8 and random.random() < intensity * 0.05:
                    remarks = [
                        "essentially",
                        "notably",
                        "importantly",
                        "significantly",
                    ]
                    remark = random.choice(remarks)
                    # Find a good place to insert - before the period
                    if modified.endswith('.'):
                        modified = modified[:-1] + f", {remark}."
                    else:
                        modified = f"{modified}, {remark}."
                
                modified_sentences.append(modified)
            
            modified_paragraphs.append(" ".join(modified_sentences))
        
        # Rejoin paragraphs with double newlines to preserve structure
        return "\n\n".join(modified_paragraphs)
    
    def vary_sentence_length(self, text):
        """Create varied sentence lengths (human pattern) while preserving paragraphs."""
        # Split by paragraphs to preserve structure
        paragraphs = text.split('\n\n')
        varied_paragraphs = []
        
        for para in paragraphs:
            sentences = sent_tokenize(para)
            varied = []
            
            i = 0
            while i < len(sentences):
                sentence = sentences[i].strip()
                
                # Occasionally combine short sentences (human pattern)
                if (i < len(sentences) - 1 and 
                    len(sentence.split()) < 8 and 
                    len(sentences[i+1].split()) < 8 and
                    random.random() < 0.3):
                    
                    next_sentence = sentences[i+1].strip()
                    connector = random.choice([", and", "; meanwhile,", ". Additionally,"])
                    
                    if sentence.endswith('.'):
                        sentence = sentence[:-1]
                    if next_sentence[0].isupper():
                        next_sentence = next_sentence[0].lower() + next_sentence[1:]
                    
                    combined = f"{sentence}{connector} {next_sentence}"
                    varied.append(combined)
                    i += 2
                    continue
                
                varied.append(sentence)
                i += 1
            
            varied_paragraphs.append(" ".join(varied))
        
        # Rejoin paragraphs with double newlines
        return "\n\n".join(varied_paragraphs)
    
    def add_uncertainty(self, text, intensity=0.3):
        """Add subtle uncertainty markers (very human) while preserving paragraphs."""
        # Split by paragraphs to preserve structure
        paragraphs = text.split('\n\n')
        modified_paragraphs = []
        
        uncertainty_markers = [
            "seems to",
            "appears to",
            "might",
            "could",
            "may",
            "tends to",
            "arguably",
        ]
        
        for para in paragraphs:
            sentences = sent_tokenize(para)
            modified = []
            
            for sentence in sentences:
                if random.random() < intensity * 0.2 and len(sentence.split()) > 5:
                    # Find a verb and add uncertainty before it
                    words = sentence.split()
                    verb_pos = None
                    
                    for j, word in enumerate(words):
                        if word and word.lower() in ['is', 'are', 'was', 'were', 'be', 'been']:
                            verb_pos = j
                            break
                    
                    if verb_pos and verb_pos > 0:
                        marker = random.choice(uncertainty_markers)
                        words.insert(verb_pos, marker)
                        sentence = " ".join(words)
                
                modified.append(sentence)
            
            modified_paragraphs.append(" ".join(modified))
        
        # Rejoin paragraphs with double newlines
        return "\n\n".join(modified_paragraphs)
    
    def humanize(self, text, intensity=0.6):
        """
        Main humanization method combining multiple techniques.
        
        Args:
            text: Input text to humanize
            intensity: Strength of humanization (0.0 to 1.0)
        
        Returns:
            Humanized text
        """
        if not text or not text.strip():
            return text
        
        result = text
        
        # Apply techniques sequentially
        result = self.add_human_variations(result, intensity)
        
        if intensity > 0.4:
            result = self.vary_sentence_length(result)
        
        if intensity > 0.6:
            result = self.add_uncertainty(result, intensity)
        
        return result
