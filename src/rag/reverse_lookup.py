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

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        """
        Calculate RRF (Reciprocal Rank Fusion) score.
        Borrowed from JapaneseHybridRetriever for consistency.

        Args:
            rank: Rank position (1-indexed)
            k: RRF constant (default: 60)

        Returns:
            RRF score
        """
        return 1.0 / (k + rank)

    def reverse_lookup(self, query: str, top_k: int = 5, config=None) -> List[ReverseLookupResult]:
        """
        Perform hybrid reverse lookup to find technical terms from descriptions.

        Uses a three-phase approach:
        1. Hybrid search (keyword + vector) with RRF fusion
        2. Confidence-based filtering
        3. LLM reranking (if needed for ambiguous cases)

        Args:
            query: Description or explanation to lookup
            top_k: Maximum number of results to return
            config: Optional RunnableConfig for LangSmith tracing

        Returns:
            List of ReverseLookupResult objects sorted by confidence
        """
        # Phase 1: Hybrid search
        keyword_results = self._keyword_search(query, top_k=10)
        vector_results = self._vector_search(query, top_k=10)

        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(keyword_results, vector_results)

        # Convert to ReverseLookupResult objects
        candidates = [
            ReverseLookupResult(
                term=term,
                confidence=score,
                source=source
            )
            for term, score, source in fused_results[:15]  # Top 15 candidates
        ]

        if not candidates:
            return []

        # Phase 2: Confidence-based filtering
        high_confidence = [c for c in candidates if c.confidence >= 0.9]
        medium_confidence = [c for c in candidates if 0.6 <= c.confidence < 0.9]

        # Case 1: High confidence with 1-2 results -> return immediately
        if 1 <= len(high_confidence) <= 2:
            logger.info(f"Reverse lookup: {len(high_confidence)} high-confidence results")
            return high_confidence

        # Case 2: Multiple high or medium confidence results -> use LLM reranking
        if len(high_confidence) >= 3 or len(medium_confidence) >= 3:
            logger.info(f"Reverse lookup: using LLM reranking for {len(candidates[:10])} candidates")
            return self._llm_rerank(query, candidates[:10], top_k, config=config)

        # Case 3: Low confidence or few results -> return as is
        return candidates[:top_k]

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Dictionary-based keyword search for reverse lookup.

        Args:
            query: Query string
            top_k: Maximum number of results

        Returns:
            List of (term, score) tuples
        """
        results = {}  # {term: score}
        query_lower = query.lower()

        # 1. Exact match (highest score)
        if query_lower in self.reverse_dict:
            for term in self.reverse_dict[query_lower]:
                results[term] = 1.0

        # 2. Pattern match (high score)
        for pattern, replacements in self.pattern_dict.items():
            if pattern in query:
                for replacement in replacements:
                    if replacement not in results or results[replacement] < 0.85:
                        results[replacement] = 0.85

        # 3. Partial match (medium score, quality-based)
        for key, terms in self.reverse_dict.items():
            if key == query_lower:  # Already handled in exact match
                continue

            if key in query_lower or query_lower in key:
                match_ratio = self._calculate_match_ratio(key, query_lower)

                if match_ratio > 0.5:  # High quality partial match only
                    score = 0.5 + (match_ratio * 0.3)  # 0.5-0.8 range

                    for term in terms:
                        if term not in results or results[term] < score:
                            results[term] = score

        # Sort by score
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _calculate_match_ratio(self, key: str, query: str) -> float:
        """Calculate the quality of a partial match"""
        longer = max(len(key), len(query))
        shorter = min(len(key), len(query))

        # Match length
        if key in query:
            match_len = len(key)
        elif query in key:
            match_len = len(query)
        else:
            # Common substring length (simplified)
            match_len = len(set(key) & set(query))

        return match_len / longer if longer > 0 else 0.0

    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Vector similarity search for jargon term definitions.

        Args:
            query: Query string
            top_k: Maximum number of results

        Returns:
            List of (term, similarity_score) tuples
        """
        if not self.vector_store:
            return []

        try:
            # Search only jargon terms (not document chunks)
            filter_dict = {
                'type': 'jargon_term',
                'collection_name': self.jargon_manager.collection_name
            }

            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_dict
            )

            term_scores = []
            for doc, distance in results:
                # Convert distance to similarity (0-1)
                similarity = 1 - distance if distance < 1 else 0
                term = doc.metadata.get('term')

                if term and similarity > 0.3:  # Minimum threshold
                    term_scores.append((term, similarity))

            return term_scores

        except Exception as e:
            logger.warning(f"Vector search for reverse lookup failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        keyword_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        k: int = 60
    ) -> List[Tuple[str, float, str]]:
        """
        Fuse keyword and vector search results using RRF.

        Args:
            keyword_results: Keyword search results [(term, score), ...]
            vector_results: Vector search results [(term, score), ...]
            k: RRF constant (default: 60)

        Returns:
            Fused results [(term, rrf_score, source), ...]
            source can be: 'keyword', 'vector', or 'hybrid'
        """
        rrf_scores = {}
        term_sources = {}  # Track which search methods found each term

        # Add RRF scores from keyword results
        for rank, (term, score) in enumerate(keyword_results, start=1):
            rrf_scores[term] = rrf_scores.get(term, 0.0) + self._rrf_score(rank, k)
            if term not in term_sources:
                term_sources[term] = set()
            term_sources[term].add('keyword')

        # Add RRF scores from vector results
        for rank, (term, score) in enumerate(vector_results, start=1):
            rrf_scores[term] = rrf_scores.get(term, 0.0) + self._rrf_score(rank, k)
            if term not in term_sources:
                term_sources[term] = set()
            term_sources[term].add('vector')

        # Normalize to 0-1 range
        if rrf_scores:
            max_score = max(rrf_scores.values())
            rrf_scores = {term: score / max_score for term, score in rrf_scores.items()}

        # Determine source label for each term
        results_with_source = []
        for term, score in rrf_scores.items():
            sources = term_sources.get(term, set())
            if len(sources) == 2:
                source_label = 'hybrid'
            elif 'keyword' in sources:
                source_label = 'keyword'
            else:
                source_label = 'vector'
            results_with_source.append((term, score, source_label))

        # Sort by score
        ranked = sorted(results_with_source, key=lambda x: x[1], reverse=True)

        return ranked

    def _llm_rerank(
        self,
        query: str,
        candidates: List[ReverseLookupResult],
        top_k: int = 5,
        config=None
    ) -> List[ReverseLookupResult]:
        """
        Rerank candidates using LLM for final selection.

        Args:
            query: Original query
            candidates: Candidate results (typically 10 or fewer)
            top_k: Number of results to return
            config: Optional RunnableConfig for LangSmith tracing

        Returns:
            Reranked results
        """
        if not self.llm:
            logger.warning("LLM not available for reranking")
            return candidates[:top_k]

        # Extract term list
        candidate_terms = [c.term for c in candidates]

        # Lightweight prompt to minimize tokens
        prompt = f"""あなたは技術文書検索の専門家です。
ユーザークエリに最も関連する専門用語を、候補の中から選んでください。

クエリ: {query}

候補用語:
{', '.join([f"{i+1}.{term}" for i, term in enumerate(candidate_terms)])}

タスク: 上記の候補から、クエリに最も関連する用語を最大{top_k}件選び、関連度の高い順に番号で答えてください。

出力形式: カンマ区切りの番号（例: 1,3,5）
回答:"""

        try:
            response = self.llm.invoke(prompt, config=config)
            selected_indices = self._parse_llm_ranking(response.content)

            # Extract selected terms
            reranked = []
            for idx in selected_indices[:top_k]:
                if 0 <= idx < len(candidates):
                    result = candidates[idx]
                    # Boost confidence slightly for LLM-selected terms
                    result.confidence = min(result.confidence + 0.1, 1.0)
                    result.source = 'llm_reranked'
                    reranked.append(result)

            return reranked if reranked else candidates[:top_k]

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return candidates[:top_k]

    def _parse_llm_ranking(self, response: str) -> List[int]:
        """Parse LLM response to extract ranking numbers"""
        try:
            # Expected format: "1,3,5" or similar
            numbers = [int(x.strip()) - 1 for x in response.strip().split(',')]  # Convert to 0-indexed
            return numbers
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []

    def _calculate_partial_match_confidence(self, key: str, query: str) -> float:
        """Calculate confidence score for partial matches (legacy method)"""
        # Longer matches get higher confidence
        match_length = len(key) if key in query else len(query)
        total_length = max(len(key), len(query))

        # Base confidence from length ratio
        confidence = match_length / total_length

        # Boost if it's a word boundary match
        if re.search(r'\b' + re.escape(key) + r'\b', query):
            confidence = min(confidence * 1.2, 0.95)

        return confidence * 0.7  # Scale down partial matches


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