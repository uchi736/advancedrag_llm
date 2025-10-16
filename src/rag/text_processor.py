import unicodedata
import re
from typing import List

# SudachiPy is no longer used (removed in favor of LLM-based processing)
SUDACHI_AVAILABLE = False

class JapaneseTextProcessor:
    """A utility class for Japanese text processing using SudachiPy."""
    
    def __init__(self):
        if SUDACHI_AVAILABLE:
            # Create SudachiPy tokenizer with mode A (shortest split)
            self.tokenizer_obj = dictionary.Dictionary().create()
            self.mode = tokenizer.Tokenizer.SplitMode.A
        else:
            self.tokenizer_obj = None
            self.mode = None
            
        # Common Japanese stop words (can be expanded)
        self.stop_words = {
            'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
            'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や',
            'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう',
            'また', 'もの', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か',
            'だ', 'これ', 'によって', 'により', 'おり', 'より', 'による', 'ず', 'なり',
            'られる', 'において', 'ば', 'なかっ', 'なく', 'しかし', 'について', 'せ', 'だっ',
            'その後', 'できる', 'それ', 'う', 'ので', 'なお', 'のみ', 'でき', 'き',
            'つ', 'における', 'および', 'いう', 'さらに', 'でも', 'ら', 'たり', 'その他',
            'に関する', 'たち', 'ます', 'ん', 'なら', 'に対して', '特に', 'せる', '及び',
            'これら', 'とき', 'では', 'にて', 'ほか', 'ながら', 'うち', 'そして', 'とも',
            'ただし', 'かつて', 'それぞれ', 'または', 'お', 'ほど', 'ものの', 'に対する',
            'ほとんど', 'と共に', 'といった', 'です', 'ました', 'ません'
        }
    
    def is_japanese(self, text: str) -> bool:
        """Checks if the text contains Japanese characters."""
        for char in text:
            name = unicodedata.name(char, '')
            if 'CJK' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                return True
        return False
    
    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """Tokenizes Japanese text using SudachiPy."""
        if not SUDACHI_AVAILABLE or not self.tokenizer_obj or not self.is_japanese(text):
            # Fallback to space-splitting for non-Japanese text
            return text.split()
        
        tokens = []
        morphemes = self.tokenizer_obj.tokenize(text, self.mode)
        
        for morpheme in morphemes:
            # Get part of speech information
            pos = morpheme.part_of_speech()[0]
            
            # Extract nouns, verbs, adjectives (can be customized)
            if pos in ['名詞', '動詞', '形容詞', '形容動詞']:
                # Use normalized form if available, otherwise use surface form
                base_form = morpheme.normalized_form()
                
                if remove_stop_words and base_form in self.stop_words:
                    continue
                    
                tokens.append(base_form)
        
        return tokens
    
    def tokenize_with_details(self, text: str) -> List[dict]:
        """Tokenizes Japanese text and returns detailed information."""
        if not SUDACHI_AVAILABLE or not self.tokenizer_obj or not self.is_japanese(text):
            return []
        
        results = []
        morphemes = self.tokenizer_obj.tokenize(text, self.mode)
        
        for morpheme in morphemes:
            pos_info = morpheme.part_of_speech()
            results.append({
                'surface': morpheme.surface(),
                'normalized': morpheme.normalized_form(),
                'reading': morpheme.reading_form(),
                'pos': pos_info[0],  # Part of speech (major category)
                'pos_detail': pos_info[1] if len(pos_info) > 1 else None,  # Sub-category
                'pos_all': pos_info  # All POS information
            })
        
        return results
    
    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """Extracts keywords (nouns and compound nouns) from Japanese text."""
        if not SUDACHI_AVAILABLE or not self.tokenizer_obj or not self.is_japanese(text):
            return []
        
        keywords = []
        morphemes = self.tokenizer_obj.tokenize(text, self.mode)
        
        # Extract nouns
        noun_sequence = []
        for i, morpheme in enumerate(morphemes):
            pos = morpheme.part_of_speech()[0]
            
            if pos == '名詞':
                noun_sequence.append(morpheme.normalized_form())
            else:
                # If we have accumulated nouns, add them as keywords
                if noun_sequence:
                    # Add individual nouns
                    for noun in noun_sequence:
                        if len(noun) >= min_length and noun not in self.stop_words:
                            keywords.append(noun)
                    
                    # Add compound noun if multiple nouns in sequence
                    if len(noun_sequence) > 1:
                        compound = ''.join(noun_sequence)
                        if len(compound) <= 10:  # Limit compound noun length
                            keywords.append(compound)
                    
                    noun_sequence = []
        
        # Handle remaining noun sequence
        if noun_sequence:
            for noun in noun_sequence:
                if len(noun) >= min_length and noun not in self.stop_words:
                    keywords.append(noun)
            
            if len(noun_sequence) > 1:
                compound = ''.join(noun_sequence)
                if len(compound) <= 10:
                    keywords.append(compound)
        
        return list(set(keywords))  # Remove duplicates
    
    def normalize_text(self, text: str) -> str:
        """
        Normalizes text, focusing on removing OCR artifacts like unwanted spaces in Japanese text.
        IMPORTANT: Preserves newlines to maintain sentence boundaries for term extraction.
        """
        # NFKC normalization (converts full-width chars to half-width)
        text = unicodedata.normalize('NFKC', text)

        # Step 1: Remove spaces around newlines FIRST (common in Azure DI output)
        # This prevents line-break artifacts like "削 減" becoming "削減"
        text = re.sub(r' *\n *', '\n', text)

        # Step 2: Normalize whitespace BUT preserve newlines
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        # Collapse multiple spaces (but not newlines) into single space
        text = re.sub(r'[ ]+', ' ', text)
        # Limit consecutive newlines to maximum of 2 (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Step 3: Remove spaces between Japanese characters (more aggressive)
        # This handles cases where words are incorrectly split like "削 減" -> "削減"
        text = re.sub(r'([ぁ-んァ-ヴー一-龠])[ ]+([ぁ-んァ-ヴー一-龠])', r'\1\2', text)

        # Step 4: Remove spaces adjacent to any Japanese character
        text = re.sub(r'([ぁ-んァ-ヴー一-龠]) ', r'\1', text)
        text = re.sub(r' ([ぁ-んァ-ヴー一-龠])', r'\1', text)

        # Step 3: Remove spaces around punctuation and brackets (but not newlines)
        text = re.sub(r' *([（(]) *', r'\1', text)
        text = re.sub(r' *([)）]) *', r'\1', text)
        text = re.sub(r' *([、。，．]) *', r'\1', text)
        text = re.sub(r' *([:]) *', r'\1', text)

        # Step 4: Remove spaces between numbers and Japanese characters (e.g., "2050 年" -> "2050年")
        text = re.sub(r'(\d) +([ぁ-んァ-ヴー一-龠])', r'\1\2', text)

        # Step 5: Collapse any remaining multiple spaces (for English parts)
        # This will not affect newlines
        text = re.sub(r'[ ]+', ' ', text)

        return text.strip()
