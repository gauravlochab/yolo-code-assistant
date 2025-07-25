"""Result ranking module for ordering search results."""

from typing import List, Dict, Any
import re


class ResultRanker:
    """Ranks search results based on various relevance factors."""
    
    def __init__(self):
        """Initialize the result ranker."""
        pass
        
    def rank_results(self, results: List[Dict[str, Any]], 
                     query: str) -> List[Dict[str, Any]]:
        """Rank search results based on relevance to query.
        
        Args:
            results: List of search results
            query: Original search query
            
        Returns:
            Ranked list of results
        """
        if not results:
            return []
            
        # Calculate relevance scores for each result
        scored_results = []
        for result in results:
            score = self._calculate_relevance_score(result, query)
            result_with_score = result.copy()
            result_with_score['relevance_score'] = score
            scored_results.append(result_with_score)
            
        # Sort by combined score (vector similarity + relevance)
        scored_results.sort(
            key=lambda x: (
                x.get('search_score', 0) * 0.7 +  # 70% weight to vector similarity
                x.get('relevance_score', 0) * 0.3  # 30% weight to other factors
            ),
            reverse=True
        )
        
        return scored_results
        
    def _calculate_relevance_score(self, result: Dict[str, Any], 
                                  query: str) -> float:
        """Calculate relevance score for a single result.
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Name match (highest weight)
        name = result.get('name', '').lower()
        if query_lower in name:
            score += 0.4
        elif any(word in name for word in query_words):
            score += 0.2
            
        # Docstring match
        docstring = (result.get('docstring') or '').lower()
        if docstring:
            if query_lower in docstring:
                score += 0.2
            elif any(word in docstring for word in query_words):
                score += 0.1
                
        # Chunk type relevance
        chunk_type = result.get('chunk_type', '')
        if 'class' in query_lower and chunk_type == 'class':
            score += 0.1
        elif 'function' in query_lower and chunk_type in ['function', 'method']:
            score += 0.1
            
        # File path relevance
        file_path = result.get('file_path', '').lower()
        if any(word in file_path for word in query_words):
            score += 0.1
            
        # Content match (lower weight to avoid noise)
        content = result.get('content', '').lower()
        if content:
            matches = len([word for word in query_words if word in content])
            score += min(0.1, matches * 0.02)
            
        return min(1.0, score)  # Cap at 1.0
        
    def filter_by_threshold(self, results: List[Dict[str, Any]], 
                           threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Filter results below a certain score threshold.
        
        Args:
            results: List of ranked results
            threshold: Minimum combined score threshold
            
        Returns:
            Filtered list of results
        """
        filtered = []
        for result in results:
            combined_score = (
                result.get('search_score', 0) * 0.7 +
                result.get('relevance_score', 0) * 0.3
            )
            if combined_score >= threshold:
                filtered.append(result)
                
        return filtered
        
    def group_by_file(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by file path.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary mapping file paths to results
        """
        grouped = {}
        for result in results:
            file_path = result.get('file_path', 'unknown')
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].append(result)
            
        # Sort results within each file by line number
        for file_path in grouped:
            grouped[file_path].sort(key=lambda x: x.get('start_line', 0))
            
        return grouped
        
    def deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar results.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of results
        """
        if not results:
            return []
            
        seen = set()
        deduplicated = []
        
        for result in results:
            # Create a unique key based on file, name, and chunk type
            key = (
                result.get('file_path', ''),
                result.get('name', ''),
                result.get('chunk_type', '')
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)
                
        return deduplicated
        
    def get_best_matches(self, results: List[Dict[str, Any]], 
                        n: int = 3) -> List[Dict[str, Any]]:
        """Get the top N best matching results.
        
        Args:
            results: List of ranked results
            n: Number of top results to return
            
        Returns:
            Top N results
        """
        return results[:n]
        
    def enhance_results_with_summary(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add summary information to results for better display.
        
        Args:
            results: List of search results
            
        Returns:
            Results with added summary information
        """
        enhanced = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Create a summary
            summary_parts = []
            
            # Add type and name
            chunk_type = result.get('chunk_type', 'code')
            name = result.get('name', 'Unknown')
            if result.get('parent_class'):
                summary_parts.append(f"{chunk_type.title()}: {result['parent_class']}.{name}")
            else:
                summary_parts.append(f"{chunk_type.title()}: {name}")
                
            # Add file location
            file_path = result.get('file_path', '')
            if file_path:
                file_name = file_path.split('/')[-1]
                lines = f"L{result.get('start_line', '?')}-{result.get('end_line', '?')}"
                summary_parts.append(f"in {file_name} {lines}")
                
            # Add docstring preview if available
            docstring = result.get('docstring', '')
            if docstring:
                preview = docstring.split('\n')[0][:80]
                if len(preview) < len(docstring):
                    preview += '...'
                summary_parts.append(f'"{preview}"')
                
            enhanced_result['summary'] = ' | '.join(summary_parts)
            enhanced.append(enhanced_result)
            
        return enhanced
