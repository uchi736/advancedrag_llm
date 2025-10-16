"""
Reverse lookup functionality for jargon terms.
Enables bidirectional search: description → technical term/abbreviation
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReverseLookupResult:
    """Result of reverse lookup operation"""
    term: str
    confidence: float
    source: str  # 'exact', 'partial', 'similarity', 'pattern', 'llm'


class ReverseLookupEngine:
    """
    Engine for reverse lookup of technical terms from descriptions.
    Combines dictionary-based reverse lookup with vector similarity search.
    """

    def __init__(self, jargon_manager=None, vector_store=None, llm=None):
        """
        Initialize the reverse lookup engine.

        Args:
            jargon_manager: JargonManager instance for dictionary access
            vector_store: PGVector store for similarity search
            llm: Language model for advanced inference
        """
        self.jargon_manager = jargon_manager
        self.vector_store = vector_store
        self.llm = llm
        self.reverse_dict = {}
        self.pattern_dict = {}

        if self.jargon_manager:
            self._build_reverse_dictionary()

    def _build_reverse_dictionary(self):
        """Build reverse lookup dictionary from existing jargon dictionary"""
        logger.info("Building reverse lookup dictionary...")

        # Get all terms from jargon manager (returns a list)
        all_terms_list = self.jargon_manager.get_all_terms()

        # Convert list to dictionary format
        all_terms = {}
        for term_data in all_terms_list:
            term = term_data.get('term')
            if term:
                all_terms[term] = term_data

        for term, info in all_terms.items():
            definition = info.get('definition', '')
            synonyms = info.get('synonyms', [])

            # Extract key phrases from definition
            key_phrases = self._extract_key_phrases(definition)

            # Add to reverse dictionary
            for phrase in key_phrases:
                phrase_lower = phrase.lower()
                if phrase_lower not in self.reverse_dict:
                    self.reverse_dict[phrase_lower] = []
                self.reverse_dict[phrase_lower].append(term)

            # Add synonyms to reverse dictionary
            for synonym in synonyms:
                if synonym:
                    synonym_lower = synonym.lower()
                    if synonym_lower not in self.reverse_dict:
                        self.reverse_dict[synonym_lower] = []
                    self.reverse_dict[synonym_lower].append(term)

        # Build pattern dictionary for common transformations
        self._build_pattern_dictionary()

        logger.info(f"Built reverse dictionary with {len(self.reverse_dict)} entries")

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from definition text"""
        key_phrases = []

        # Extract parenthetical explanations (e.g., "亜酸化窒素（N2O）")
        parenthetical = re.findall(r'（([^）]+)）|\(([^)]+)\)', text)
        for match in parenthetical:
            phrase = match[0] if match[0] else match[1]
            if phrase and len(phrase) > 1:
                key_phrases.append(phrase)

        # Extract the first noun phrase (often the main definition)
        first_phrase_match = re.match(r'^([^、。，]+)[、。，]', text)
        if first_phrase_match:
            phrase = first_phrase_match.group(1).strip()
            # Remove "とは" and similar
            phrase = re.sub(r'とは$|は$|です$|である$', '', phrase)
            if phrase and len(phrase) > 2:
                key_phrases.append(phrase)

        # Extract characteristic numbers (e.g., "265倍")
        numbers = re.findall(r'\d+倍|\d+％|\d+%', text)
        key_phrases.extend(numbers)

        return key_phrases

    def _build_pattern_dictionary(self):
        """Build common pattern transformations"""
        self.pattern_dict = {
            # Japanese to English abbreviations
            '亜酸化窒素': ['N2O', 'nitrous oxide'],
            '二酸化炭素': ['CO2', 'carbon dioxide'],
            '一酸化炭素': ['CO', 'carbon monoxide'],
            '窒素酸化物': ['NOx', 'nitrogen oxides'],
            '硫黄酸化物': ['SOx', 'sulfur oxides'],
            '温室効果ガス': ['GHG', 'greenhouse gas'],
            '国際海事機関': ['IMO', 'International Maritime Organization'],
            'アンモニア': ['NH3', 'ammonia'],
            '水素': ['H2', 'hydrogen'],
            'メタン': ['CH4', 'methane'],
            # Common technical terms
            '削減': ['reduction', 'mitigation'],
            '排出': ['emission', 'discharge'],
            '燃焼': ['combustion', 'burning'],
            '効率': ['efficiency'],
            '環境影響': ['environmental impact'],
        }

    def reverse_lookup(self, query: str, top_k: int = 5) -> List[ReverseLookupResult]:
        """
        Perform reverse lookup to find technical terms from descriptions.

        Args:
            query: Description or explanation to lookup
            top_k: Maximum number of results to return

        Returns:
            List of ReverseLookupResult objects sorted by confidence
        """
        results = []
        seen_terms = set()

        # Step 1: Exact match in reverse dictionary
        query_lower = query.lower()
        if query_lower in self.reverse_dict:
            for term in self.reverse_dict[query_lower]:
                if term not in seen_terms:
                    results.append(ReverseLookupResult(
                        term=term,
                        confidence=1.0,
                        source='exact'
                    ))
                    seen_terms.add(term)

        # Step 2: Partial match in reverse dictionary
        for key, terms in self.reverse_dict.items():
            if key in query_lower or query_lower in key:
                for term in terms:
                    if term not in seen_terms:
                        # Calculate confidence based on match quality
                        confidence = self._calculate_partial_match_confidence(key, query_lower)
                        results.append(ReverseLookupResult(
                            term=term,
                            confidence=confidence,
                            source='partial'
                        ))
                        seen_terms.add(term)

        # Step 3: Pattern-based lookup
        for pattern, replacements in self.pattern_dict.items():
            if pattern in query:
                for replacement in replacements:
                    if replacement not in seen_terms:
                        results.append(ReverseLookupResult(
                            term=replacement,
                            confidence=0.8,
                            source='pattern'
                        ))
                        seen_terms.add(replacement)

        # Step 4: Vector similarity search (if available)
        if self.vector_store:
            similar_results = self._similarity_search(query, top_k=10)
            for term, score in similar_results:
                if term not in seen_terms:
                    results.append(ReverseLookupResult(
                        term=term,
                        confidence=score,
                        source='similarity'
                    ))
                    seen_terms.add(term)

        # Sort by confidence and return top_k
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def _calculate_partial_match_confidence(self, key: str, query: str) -> float:
        """Calculate confidence score for partial matches"""
        # Longer matches get higher confidence
        match_length = len(key) if key in query else len(query)
        total_length = max(len(key), len(query))

        # Base confidence from length ratio
        confidence = match_length / total_length

        # Boost if it's a word boundary match
        if re.search(r'\b' + re.escape(key) + r'\b', query):
            confidence = min(confidence * 1.2, 0.95)

        return confidence * 0.7  # Scale down partial matches

    def _similarity_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search for related terms.

        Returns:
            List of (term, similarity_score) tuples
        """
        try:
            # Search in vector store with metadata filter for jargon terms
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter={"type": "jargon_term"}
            )

            # Extract terms and scores
            term_scores = []
            for doc, score in results:
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - score if score < 1 else 0
                term = doc.metadata.get('term', doc.page_content[:50])
                term_scores.append((term, similarity))

            return term_scores

        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []

    def augment_query_with_reverse_lookup(
        self,
        original_query: str,
        extracted_terms: List[str]
    ) -> Dict[str, any]:
        """
        Augment query using both forward and reverse lookup.

        Args:
            original_query: Original user query
            extracted_terms: Terms extracted from query (forward lookup)

        Returns:
            Dictionary with augmentation details
        """
        augmentation = {
            'original_query': original_query,
            'forward_terms': extracted_terms,
            'reverse_terms': [],
            'combined_terms': [],
            'augmented_queries': []
        }

        # Perform reverse lookup
        reverse_results = self.reverse_lookup(original_query, top_k=5)
        reverse_terms = [(r.term, r.confidence) for r in reverse_results]
        augmentation['reverse_terms'] = reverse_terms

        # Combine forward and reverse terms
        all_terms = set(extracted_terms)
        all_terms.update([t for t, _ in reverse_terms if _ > 0.5])  # Only high confidence
        augmentation['combined_terms'] = list(all_terms)

        # Generate multiple query patterns
        queries = [original_query]  # Original

        # Add term-expanded query
        if all_terms:
            term_expansion = f"{original_query} ({' OR '.join(all_terms)})"
            queries.append(term_expansion)

        # Add weighted query for high-confidence terms
        weighted_parts = []
        for term, conf in reverse_terms[:3]:  # Top 3 reverse lookup results
            if conf > 0.7:
                weighted_parts.append(f"{term}^{conf:.1f}")

        if weighted_parts:
            weighted_query = f"{original_query} {' '.join(weighted_parts)}"
            queries.append(weighted_query)

        augmentation['augmented_queries'] = queries

        return augmentation