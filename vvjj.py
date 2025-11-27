"""
madl_engine.optimized_search
----------------------------

Full rewrite of the MADL optimized search pipeline to use:

 - Azure OpenAI (certificate auth) for embeddings (text-embedding-ada-002)
 - OpenSearch KNN (FAISS/HNSW) as the vector store
 - All previous MADL normalization, grouping and ranking logic

Provides:
 - _global_pipeline : OptimizedSearchPipeline instance for import by core.py
 - gherkin_processor : convenience wrapper for BDD -> grouped search
"""

from __future__ import annotations
import re
import time
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

# Optional libraries (used only if available)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False

# Local helpers (these modules should exist in your project)
from madl_engine.azure_embeddings import generate_embedding
from madl_engine.opensearch_client import get_os_client

logger = logging.getLogger("madl_optimized_search")
logger.setLevel(logging.INFO)


# -------------------------
# Simple SearchResult class
# -------------------------
@dataclass
class SearchResult:
    score: float
    payload: Dict[str, Any]


# -------------------------
# Configuration dataclass
# -------------------------
@dataclass
class SearchConfig:
    min_score: float = 0.6
    top_k: int = 10
    rerank_top_k: int = 5

    # Feature flags
    use_normalization: bool = True
    use_query_expansion: bool = True
    use_keyword_boosting: bool = True
    use_reranking: bool = False
    use_multi_level: bool = True
    use_context_aware: bool = True
    use_fuzzy_matching: bool = True

    # Caching
    cache_embeddings: bool = True
    cache_results: bool = True
    cache_ttl: int = 3600  # seconds


# -------------------------
# Query Normalizer + Expander
# -------------------------
class QueryNormalizer:
    GHERKIN_KEYWORDS = ['given', 'when', 'then', 'and', 'but', 'i']
    STOP_WORDS = ['a', 'an', 'the', 'with', 'on', 'in', 'at', 'to', 'for', 'of']

    SYNONYMS = {
        'click': ['press', 'tap', 'select', 'hit'],
        'enter': ['input', 'type', 'fill', 'provide'],
        'login': ['authenticate', 'sign in', 'log in'],
        'logout': ['sign out', 'log out'],
        'verify': ['check', 'validate', 'assert', 'confirm', 'ensure'],
        'navigate': ['go to', 'open', 'visit', 'move to', 'goto'],
        'create': ['add', 'new', 'insert'],
        'delete': ['remove', 'clear'],
        'update': ['modify', 'change', 'edit'],
        'save': ['submit', 'store'],
        'button': ['btn'],
        'username': ['user', 'email'],
        'password': ['pwd', 'pass'],
        'page': ['screen', 'view'],
        'field': ['textbox', 'inputfield'],
        'visible': ['displayed', 'shown'],
        'select': ['choose', 'pick'],
    }

    SYNONYM_REVERSE = {}
    for k, vals in SYNONYMS.items():
        for v in vals:
            SYNONYM_REVERSE[v] = k

    @classmethod
    def normalize(cls, text: str, is_gherkin: bool = False) -> str:
        if not text:
            return ""

        text = text.lower().strip()
        # break identifiers
        text = text.replace('_', ' ').replace('::', ' ').replace('->', ' ').replace('.', ' ')
        params = re.findall(r'"([^"]+)"', text)
        text = re.sub(r'"[^"]+"', '', text)
        words = text.split()

        if is_gherkin:
            words = [w for w in words if w not in cls.GHERKIN_KEYWORDS]

        if len(words) > 3:
            words = [w for w in words if w not in cls.STOP_WORDS]

        normalized_words = []
        for w in words:
            normalized_words.append(cls.SYNONYM_REVERSE.get(w, w))

        normalized_words = [re.sub(r'[^a-z0-9]', '', w) for w in normalized_words]
        normalized_words = [w for w in normalized_words if w]

        seen = set()
        dedup = []
        for w in normalized_words:
            if w not in seen:
                seen.add(w)
                dedup.append(w)

        result = ' '.join(dedup)
        if params:
            param_text = ' '.join([p.lower().replace('@', 'at').replace('.', 'dot') for p in params])
            result = f"{result} {param_text}"

        return result.strip()

    @classmethod
    def expand_query(cls, text: str) -> List[str]:
        queries = [text]
        words = text.lower().split()
        for i, w in enumerate(words):
            if w in cls.SYNONYMS:
                for synonym in cls.SYNONYMS[w][:2]:
                    variant = words.copy()
                    variant[i] = synonym
                    queries.append(' '.join(variant))
        return queries[:4]


# -------------------------
# Keyword Booster
# -------------------------
class KeywordBooster:
    WEIGHTS = {
        'login': 2.0, 'logout': 2.0, 'verify': 1.8, 'validate': 1.8,
        'create': 1.7, 'delete': 1.7, 'save': 1.6, 'submit': 1.6,
        'button': 1.5, 'link': 1.5, 'field': 1.4, 'input': 1.4, 'dropdown': 1.4,
        'click': 1.3, 'enter': 1.3, 'select': 1.3,
        'username': 1.2, 'password': 1.2, 'email': 1.2, 'name': 1.1
    }

    @classmethod
    def boost_query(cls, text: str) -> str:
        words = text.split()
        boosted = []
        for word in words:
            weight = cls.WEIGHTS.get(word.lower(), 1.0)
            repetitions = int(weight)
            boosted.extend([word] * max(repetitions, 1))
        return ' '.join(boosted)


# -------------------------
# Gherkin Parser (unchanged)
# -------------------------
class GherkinStepParser:
    ACTION_CATEGORIES = {
        'setup': ['given', 'i am', 'i have', 'user is'],
        'action': ['when', 'i click', 'i enter', 'i select', 'i submit'],
        'verification': ['then', 'i should', 'verify', 'check', 'assert'],
        'navigation': ['navigate', 'go to', 'open', 'visit'],
        'form': ['enter', 'input', 'fill', 'type', 'select']
    }

    def get_action_category(self, step_text: str) -> str:
        t = step_text.lower()
        for cat, pats in self.ACTION_CATEGORIES.items():
            if any(p in t for p in pats):
                return cat
        return 'other'

    def extract_action(self, step_text: str) -> Tuple[str, str]:
        keywords = ['Given', 'When', 'Then', 'And', 'But']
        for kw in keywords:
            if step_text.strip().startswith(kw):
                return kw, step_text[len(kw):].strip()
        return '', step_text

    def group_related_steps(self, steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not steps:
            return []
        groups = []
        current = [steps[0]]
        def is_cont(text): return text.strip().lower().startswith(("and ", "but "))
        for i in range(1, len(steps)):
            curr = steps[i]["step_text"]
            if is_cont(curr):
                current.append(steps[i])
            else:
                groups.append(current)
                current = [steps[i]]
        groups.append(current)
        return groups

    def create_combined_search_query(self, steps_group: List[Dict[str, Any]]) -> str:
        texts = [self.extract_action(s["step_text"])[1].lower() for s in steps_group]
        if len(texts) == 1:
            return texts[0]
        combined = " ".join(texts)
        if ("username" in combined and "password" in combined) or ("user name" in combined and "password" in combined):
            return "login with username and password"
        if all(("enter" in t or "input" in t) for t in texts):
            cleaned = [t.replace("enter", "").replace("input", "").strip() for t in texts]
            return "fill form: " + " and ".join(cleaned)
        if "click" in combined and any("enter" in t for t in texts):
            return "fill form and click button"
        return " then ".join(texts)


# -------------------------
# Context Scorer
# -------------------------
class ContextAwareScorer:
    def __init__(self):
        self.page_keywords = {
            'LoginPage': ['login', 'signin', 'authenticate', 'credentials'],
            'ContactPage': ['contact', 'address', 'phone', 'email'],
            'DashboardPage': ['dashboard', 'home', 'overview'],
            'CheckoutPage': ['checkout', 'payment', 'order', 'cart'],
        }

    def score_with_context(self, method: Dict[str, Any], context: Dict[str, Any]) -> float:
        score = 0.0
        current_page = context.get('current_page') if context else None
        if current_page:
            method_class = method.get('class_name', '')
            if current_page.lower() in method_class.lower():
                score += 0.3
            if current_page in self.page_keywords:
                method_text = f"{method.get('intent','')} {' '.join(method.get('keywords',[]))}".lower()
                page_keywords = self.page_keywords[current_page]
                matches = sum(1 for kw in page_keywords if kw in method_text)
                score += (matches / len(page_keywords)) * 0.2

        previous_steps = context.get('previous_steps', []) if context else []
        if previous_steps:
            for prev in previous_steps:
                for page_class, keywords in self.page_keywords.items():
                    if any(kw in prev.lower() for kw in keywords):
                        if page_class.lower() in method.get('class_name','').lower():
                            score += 0.1
                            break
        return min(score, 1.0)


# -------------------------
# Multi-Level Search Engine (OpenSearch-backed)
# -------------------------
class MultiLevelSearchEngine:
    def __init__(self, config: SearchConfig = None, index_name: str = "madl_methods_v2"):
        self.config = config or SearchConfig()
        self.index_name = index_name
        self.normalizer = QueryNormalizer()
        self.booster = KeywordBooster()
        self.context_scorer = ContextAwareScorer()
        self.os_client = get_os_client()

        # optional components
        self.reranker = None
        if self.config.use_reranking and CROSS_ENCODER_AVAILABLE:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder loaded")
            except Exception as e:
                logger.warning(f"Cross-encoder load failed: {e}")

        # caches
        self.embedding_cache: Dict[str, List[float]] = {}
        self.result_cache: Dict[str, Tuple[List[SearchResult], float]] = {}

    # caching helper
    @lru_cache(maxsize=2048)
    def _cached_embedding(self, text_hash: str, text: str):
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        emb = generate_embedding(text)
        if self.config.cache_embeddings:
            self.embedding_cache[text_hash] = emb
        return emb

    def _get_embedding(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        return self._cached_embedding(key, text)

    def _os_knn_search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        # Build OpenSearch KNN query body
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        resp = self.os_client.search(index=self.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        results = []
        for h in hits:
            # OpenSearch returns normalized score under _score when using script_score/knn
            score = float(h.get("_score", 0.0))
            source = h.get("_source", {})
            results.append(SearchResult(score=score, payload=source))
        return results

    def level1_search(self, query: str, top_k: int, min_score: float,
                      context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        # normalize & boost
        q = self.normalizer.normalize(query) if self.config.use_normalization else query
        if self.config.use_keyword_boosting:
            q = self.booster.boost_query(q)

        # embedding
        vector = self._get_embedding(q)

        # run knn
        raw_results = self._os_knn_search(vector, top_k)

        # filter by min_score threshold
        filtered = [r for r in raw_results if r.score >= min_score]

        # context boosting
        if self.config.use_context_aware and context:
            for r in filtered:
                try:
                    method_payload = r.payload
                    cscore = self.context_scorer.score_with_context(method_payload, context)
                    r.score = r.score * 0.9 + cscore * 0.1
                except Exception:
                    pass
            filtered.sort(key=lambda x: x.score, reverse=True)

        # optional fuzzy boost
        if self.config.use_fuzzy_matching and FUZZY_AVAILABLE:
            qnorm = self.normalizer.normalize(query)
            for r in filtered:
                name = r.payload.get("method_name", "").lower()
                fuzzy_score = fuzz.ratio(qnorm, name) / 100.0
                r.score = r.score * 0.95 + fuzzy_score * 0.05
            filtered.sort(key=lambda x: x.score, reverse=True)

        return filtered[:top_k]

    def search_with_query_expansion(self, query: str, top_k: int, min_score: float, context: Optional[Dict] = None) -> List[SearchResult]:
        variants = self.normalizer.expand_query(query)
        all_results: Dict[str, SearchResult] = {}
        for variant in variants:
            res = self.level1_search(variant, top_k, min_score * 0.9, context)
            for r in res:
                mid = f"{r.payload.get('class_name','')}.{r.payload.get('method_name','')}"
                if mid not in all_results or r.score > all_results[mid].score:
                    all_results[mid] = r
        merged = list(all_results.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def rerank_with_cross_encoder(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not self.reranker or not results:
            return results
        pairs = []
        for r in results:
            method_text = f"{r.payload.get('method_name','')} {r.payload.get('intent','')} {r.payload.get('semantic_description','')}"
            pairs.append([query, method_text])
        try:
            ce_scores = self.reranker.predict(pairs)
            for i, r in enumerate(results):
                r.score = 0.6 * r.score + 0.4 * float(ce_scores[i])
            results.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
        return results

    def optimized_search(self, query: str, top_k: int = 5, min_score: float = 0.6, context: Optional[Dict] = None) -> List[SearchResult]:
        cache_key = f"{query}|{top_k}|{min_score}|{context}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        if self.config.cache_results and cache_hash in self.result_cache:
            cached, ts = self.result_cache[cache_hash]
            if time.time() - ts < self.config.cache_ttl:
                return cached

        # Level 1
        if self.config.use_query_expansion:
            results = self.search_with_query_expansion(query, top_k * 2, min_score, context)
        else:
            results = self.level1_search(query, top_k * 2, min_score, context)

        if not results:
            if self.config.cache_results:
                self.result_cache[cache_hash] = ([], time.time())
            return []

        # optionally rerank
        if self.config.use_reranking and self.reranker:
            top_for_rerank = results[: self.config.rerank_top_k * 2]
            results = self.rerank_with_cross_encoder(query, top_for_rerank) + results[self.config.rerank_top_k * 2 :]

        # final sort & slice
        results.sort(key=lambda x: x.score, reverse=True)
        final = results[:top_k]

        if self.config.cache_results:
            self.result_cache[cache_hash] = (final, time.time())

        return final


# -------------------------
# Gherkin Processor wrapper
# -------------------------
class GherkinSearchProcessor:
    def __init__(self, search_pipeline: OptimizedSearchPipeline := None):
        # We'll initialize pipeline lazily to avoid import cycles in some setups
        self.pipeline = search_pipeline or OptimizedSearchPipeline()
        self.parser = GherkinStepParser()

    def _prepare_step_objects(self, steps: List[str]) -> List[Dict[str, Any]]:
        return [{"index": i + 1, "step_text": s.strip()} for i, s in enumerate(steps)]

    def group_steps(self, steps: List[str]) -> List[List[Dict[str, Any]]]:
        step_objs = self._prepare_step_objects(steps)
        return self.parser.group_related_steps(step_objs)

    def build_query_for_group(self, group: List[Dict[str, Any]]) -> str:
        return self.parser.create_combined_search_query(group)

    def analyze(self, steps: List[str], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        final_results = []
        if not steps:
            return final_results

        groups = self.group_steps(steps)
        for g in groups:
            start = g[0]["index"]
            end = g[-1]["index"]
            label = f"{start}_{end}"
            details = "\n".join([s["step_text"] for s in g])
            query = self.build_query_for_group(g)
            matches_raw = self.pipeline.search(query, top_k=5, min_score=0.5, context=context)
            matches = []
            for m in matches_raw:
                payload = m.payload
                matches.append({
                    "method_name": payload.get("method_name", ""),
                    "class_name": payload.get("class_name", ""),
                    "signature": payload.get("full_signature", payload.get("signature", "")),
                    "method_code": payload.get("method_code", ""),
                    "score": round(float(m.score) * 100, 2)
                })
            final_results.append({
                "step_label": label,
                "step_details": details,
                "query": query,
                "matches": matches
            })
        return final_results


# -------------------------
# Adapter / Pipeline class to keep external API stable
# -------------------------
class OptimizedSearchPipeline:
    def __init__(self, config: Optional[SearchConfig] = None, index_name: str = "madl_methods_v2"):
        self.config = config or SearchConfig()
        self.engine = MultiLevelSearchEngine(self.config, index_name)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.6, context: Optional[Dict] = None) -> List[SearchResult]:
        return self.engine.optimized_search(query, top_k, min_score, context)

    def clear_cache(self):
        self.engine.result_cache.clear()
        self.engine.embedding_cache.clear()
        self.engine._cached_embedding.cache_clear()


# -------------------------
# Instantiate a global pipeline and processor for import by core.py
# -------------------------
_global_pipeline = OptimizedSearchPipeline()
gherkin_processor = GherkinSearchProcessor(_global_pipeline)
