"""
madl_engine.optimized_search (Enhanced - Option 2)

Features:
 - Azure OpenAI embeddings (via azure_embeddings.generate_embedding)
 - OpenSearch KNN (vector) search + full-text fallback
 - Query normalization, expansion, keyword boosting
 - Context-aware scoring
 - Cross-encoder reranking (optional)
 - Fuzzy tie-breaking (optional)
 - Caching and deduplication
 - Gherkin grouping and pipeline wrapper

Author: Generated for your JPM environment
"""

from __future__ import annotations
import re
import time
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

# Attempt to import optional libs
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

# Local helpers you must provide
from madl_engine.azure_embeddings import generate_embedding
from madl_engine.opensearch_client import get_os_client

logger = logging.getLogger("madl_optimized_search")
logger.setLevel(logging.INFO)


# -------------------------
# Lightweight wrapper for search results
# -------------------------
@dataclass
class SearchResult:
    score: float
    payload: Dict[str, Any]


# -------------------------
# Configuration
# -------------------------
@dataclass
class SearchConfig:
    # retrieval params
    min_score: float = 0.5
    top_k: int = 10
    text_fallback_k: int = 50

    # feature flags
    use_normalization: bool = True
    use_query_expansion: bool = True
    use_keyword_boosting: bool = True
    use_reranking: bool = False
    use_context_aware: bool = True
    use_fuzzy_matching: bool = True
    use_text_fallback: bool = True

    # reranking / cross-encoder
    rerank_top_k: int = 5

    # fusion weights
    vector_weight: float = 0.75
    text_weight: float = 0.25

    # caching
    cache_embeddings: bool = True
    cache_results: bool = True
    cache_ttl: int = 3600  # seconds

    # OpenSearch index name
    index_name: str = "madl_methods_v2"

    # embedding dimension (ADA-002)
    embedding_dim: int = 1536


# -------------------------
# Query Normalizer + Expander
# -------------------------
class QueryNormalizer:
    GHERKIN_KEYWORDS = {'given', 'when', 'then', 'and', 'but', 'i'}
    STOP_WORDS = {'a', 'an', 'the', 'with', 'on', 'in', 'at', 'to', 'for', 'of'}

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

    # reverse map for quick normalization
    SYNONYM_REVERSE = {}
    for k, vals in SYNONYMS.items():
        for v in vals:
            SYNONYM_REVERSE[v] = k

    @classmethod
    def normalize(cls, text: str, is_gherkin: bool = False) -> str:
        if not text:
            return ""
        text = text.lower().strip()

        # break compound identifiers
        text = text.replace('_', ' ').replace('::', ' ').replace('->', ' ').replace('.', ' ')

        # pull out quoted params (we'll append them back)
        params = re.findall(r'"([^"]+)"', text)
        text = re.sub(r'"[^"]+"', '', text)

        words = text.split()
        if is_gherkin:
            words = [w for w in words if w not in cls.GHERKIN_KEYWORDS]

        if len(words) > 3:
            words = [w for w in words if w not in cls.STOP_WORDS]

        normalized = []
        for w in words:
            w2 = cls.SYNONYM_REVERSE.get(w, w)
            w2 = re.sub(r'[^a-z0-9]', '', w2)
            if w2:
                normalized.append(w2)

        # deduplicate preserving order
        seen = set()
        dedup = []
        for w in normalized:
            if w not in seen:
                seen.add(w)
                dedup.append(w)

        base = " ".join(dedup)
        if params:
            params_norm = " ".join([p.lower().replace('@', 'at').replace('.', 'dot') for p in params])
            base = f"{base} {params_norm}"
        return base.strip()

    @classmethod
    def expand_query(cls, text: str) -> List[str]:
        queries = [text]
        words = text.lower().split()
        for i, w in enumerate(words):
            if w in cls.SYNONYMS:
                for syn in cls.SYNONYMS[w][:2]:
                    variant = words.copy()
                    variant[i] = syn
                    queries.append(" ".join(variant))
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
        tokens = text.split()
        boosted = []
        for t in tokens:
            w = cls.WEIGHTS.get(t.lower(), 1.0)
            repetitions = int(w)
            boosted.extend([t] * max(repetitions, 1))
        return " ".join(boosted)


# -------------------------
# Gherkin Step Parser (unchanged, robust)
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
        for k, pats in self.ACTION_CATEGORIES.items():
            if any(p in t for p in pats):
                return k
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
        def is_cont(x): return x.strip().lower().startswith(("and ", "but "))
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
# Context-aware scorer
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
        if not context:
            return score

        current_page = context.get('current_page')
        if current_page:
            method_class = method.get('class_name', '')
            if current_page.lower() in method_class.lower():
                score += 0.35

            if current_page in self.page_keywords:
                method_text = f"{method.get('intent','')} {' '.join(method.get('keywords',[]))}".lower()
                page_keys = self.page_keywords[current_page]
                matches = sum(1 for kw in page_keys if kw in method_text)
                if page_keys:
                    score += (matches / len(page_keys)) * 0.25

        # previous steps context
        previous_steps = context.get('previous_steps', []) or []
        for prev in previous_steps:
            for page_class, keys in self.page_keywords.items():
                if any(k in prev.lower() for k in keys):
                    if page_class.lower() in method.get('class_name', '').lower():
                        score += 0.1
                        break

        return min(score, 1.0)


# -------------------------
# Multi-level search engine (OpenSearch + hybrid)
# -------------------------
class MultiLevelSearchEngine:
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.normalizer = QueryNormalizer()
        self.booster = KeywordBooster()
        self.context_scorer = ContextAwareScorer()
        self.os_client = get_os_client()

        # cross-encoder
        self.reranker = None
        if self.config.use_reranking and CROSS_ENCODER_AVAILABLE:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder loaded.")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")

        # caches
        self.embedding_cache: Dict[str, List[float]] = {}
        self.result_cache: Dict[str, Tuple[List[SearchResult], float]] = {}

    # ---- embedding helpers ----
    @lru_cache(maxsize=2048)
    def _cached_embedding(self, key: str, text: str) -> List[float]:
        if key in self.embedding_cache:
            return self.embedding_cache[key]
        emb = generate_embedding(text)
        if self.config.cache_embeddings:
            self.embedding_cache[key] = emb
        return emb

    def _get_embedding(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        return self._cached_embedding(key, text)

    # ---- OpenSearch KNN vector search ----
    def _os_knn_search(self, vector: List[float], k: int) -> List[SearchResult]:
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": k
                    }
                }
            }
        }
        resp = self.os_client.search(index=self.config.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        results: List[SearchResult] = []
        for h in hits:
            score = float(h.get("_score", 0.0))
            payload = h.get("_source", {})
            results.append(SearchResult(score=score, payload=payload))
        return results

    # ---- OpenSearch text fallback (BM25 match) ----
    def _os_text_search(self, query: str, k: int) -> List[SearchResult]:
        body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["intent^3", "semantic_description^2", "method_name^2", "method_code", "full_signature"],
                    "type": "best_fields",
                    "operator": "and"
                }
            }
        }
        resp = self.os_client.search(index=self.config.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        results = []
        for h in hits:
            score = float(h.get("_score", 0.0))
            payload = h.get("_source", {})
            results.append(SearchResult(score=score, payload=payload))
        return results

    # ---- normalize scores to 0..1 per list ----
    @staticmethod
    def _normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            for r in results:
                r.score = 1.0
            return results
        for r in results:
            # min-max normalize
            r.score = (r.score - min_s) / (max_s - min_s)
        return results

    # ---- fuse vector + text results (by doc id) ----
    def _fuse_results(self, vector_results: List[SearchResult], text_results: List[SearchResult]) -> List[SearchResult]:
        # normalize both sets
        v_norm = self._normalize_scores(vector_results)
        t_norm = self._normalize_scores(text_results)

        merged: Dict[str, SearchResult] = {}
        # helper to id a doc
        def doc_id(payload):
            return f"{payload.get('class_name','')}.{payload.get('method_name','')}"

        for r in v_norm:
            merged[doc_id(r.payload)] = SearchResult(score=r.score * self.config.vector_weight, payload=r.payload)

        for r in t_norm:
            did = doc_id(r.payload)
            if did in merged:
                merged[did].score += r.score * self.config.text_weight
            else:
                merged[did] = SearchResult(score=r.score * self.config.text_weight, payload=r.payload)

        final = list(merged.values())
        final.sort(key=lambda x: x.score, reverse=True)
        return final

    # ---- apply context boosting ----
    def _apply_context(self, results: List[SearchResult], context: Optional[Dict[str, Any]]) -> List[SearchResult]:
        if not context or not results:
            return results
        for r in results:
            try:
                bonus = self.context_scorer.score_with_context(r.payload, context)
                r.score = r.score * 0.9 + bonus * 0.1
            except Exception:
                pass
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # ---- fuzzy tie-breaker ----
    def _apply_fuzzy(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not FUZZY_AVAILABLE or not results:
            return results
        qnorm = self.normalizer.normalize(query)
        for r in results:
            name = r.payload.get("method_name", "").lower()
            fuzzy_score = fuzz.ratio(qnorm, name) / 100.0
            r.score = r.score * 0.95 + fuzzy_score * 0.05
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # ---- rerank with cross-encoder ----
    def _rerank_with_cross_encoder(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not self.reranker or not results:
            return results
        pairs = []
        for r in results:
            method_text = f"{r.payload.get('method_name','')} {r.payload.get('intent','')} {r.payload.get('semantic_description','')}"
            pairs.append([query, method_text])
        try:
            ce_scores = self.reranker.predict(pairs)
            for i, r in enumerate(results):
                # combine previous score + ce_score
                r.score = 0.6 * r.score + 0.4 * float(ce_scores[i])
            results.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed: {e}")
        return results

    # ---- deduplicate by doc id preserving top score ----
    def _dedupe(self, results: List[SearchResult]) -> List[SearchResult]:
        seen = {}
        final = []
        for r in results:
            did = f"{r.payload.get('class_name','')}.{r.payload.get('method_name','')}"
            if did not in seen:
                seen[did] = r
            else:
                if r.score > seen[did].score:
                    seen[did] = r
        # maintain sorted order by score
        final = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return final

    # ---- public search API ----
    def optimized_search(self, query: str, top_k: int = 5, min_score: float = 0.0, context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Full search pipeline:
          - normalize / boost / expand
          - embed query
          - vector KNN search
          - (optional) text fallback search
          - fuse vector+text scores
          - context boost, rerank, fuzzy, dedupe
          - return top_k
        """

        cache_key_raw = f"{query}|{top_k}|{min_score}|{context}"
        cache_hash = hashlib.md5(cache_key_raw.encode()).hexdigest()
        if self.config.cache_results and cache_hash in self.result_cache:
            results_cached, ts = self.result_cache[cache_hash]
            if time.time() - ts < self.config.cache_ttl:
                return results_cached

        # normalization & keyword boosting
        q_norm = self.normalizer.normalize(query, is_gherkin=False) if self.config.use_normalization else query
        if self.config.use_keyword_boosting:
            q_boosted = self.booster.boost_query(q_norm)
        else:
            q_boosted = q_norm

        # embedding
        emb_text = q_boosted
        emb_vector = self._get_embedding(emb_text)

        # vector search
        vector_hits = self._os_knn_search(emb_vector, k=max(top_k * 3, self.config.text_fallback_k))

        # optional text fallback / expansion
        if self.config.use_text_fallback:
            # query expansion helps recall
            if self.config.use_query_expansion:
                variants = self.normalizer.expand_query(q_boosted)
                text_hits_acc: Dict[str, SearchResult] = {}
                for var in variants:
                    th = self._os_text_search(var, k=self.config.text_fallback_k)
                    for t in th:
                        did = f"{t.payload.get('class_name','')}.{t.payload.get('method_name','')}"
                        # keep highest text score for each id
                        if did not in text_hits_acc or t.score > text_hits_acc[did].score:
                            text_hits_acc[did] = t
                text_hits = list(text_hits_acc.values())
            else:
                text_hits = self._os_text_search(q_boosted, k=self.config.text_fallback_k)
        else:
            text_hits = []

        # fuse results
        if vector_hits and text_hits:
            fused = self._fuse_results(vector_hits, text_hits)
        elif vector_hits:
            fused = self._normalize_scores(vector_hits)
        else:
            fused = self._normalize_scores(text_hits)

        # apply context boost
        if self.config.use_context_aware and context:
            fused = self._apply_context(fused, context)

        # rerank if enabled
        if self.config.use_reranking and self.reranker:
            fused_top = fused[: self.config.rerank_top_k]
            fused_top = self._rerank_with_cross_encoder(q_boosted, fused_top)
            fused = fused_top + fused[self.config.rerank_top_k :]

        # fuzzy tie-breaker
        if self.config.use_fuzzy_matching:
            fused = self._apply_fuzzy(q_boosted, fused)

        # dedupe
        fused = self._dedupe(fused)

        # filter min_score threshold
        final_filtered = [r for r in fused if r.score >= min_score] if min_score > 0 else fused

        # sort & truncate to top_k
        final_filtered.sort(key=lambda x: x.score, reverse=True)
        final = final_filtered[:top_k]

        if self.config.cache_results:
            self.result_cache[cache_hash] = (final, time.time())

        return final


# -------------------------
# GherkinSearchProcessor wrapper
# -------------------------
class GherkinSearchProcessor:
    """
    Groups BDD steps -> builds combined query -> calls search pipeline
    """

    def __init__(self, pipeline: Optional[OptimizedSearchPipeline] = None):
        from typing import TYPE_CHECKING
        # lazy import of pipeline class (to avoid circular)
        self.parser = GherkinStepParser()
        self.pipeline = pipeline or OptimizedSearchPipeline()

    def _prepare_step_objects(self, steps: List[str]) -> List[Dict[str, Any]]:
        return [{"index": i + 1, "step_text": s.strip()} for i, s in enumerate(steps)]

    def group_steps(self, steps: List[str]) -> List[List[Dict[str, Any]]]:
        step_objs = self._prepare_step_objects(steps)
        return self.parser.group_related_steps(step_objs)

    def build_query_for_group(self, group: List[Dict[str, Any]]) -> str:
        return self.parser.create_combined_search_query(group)

    def analyze(self, steps: List[str], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        results_out = []
        if not steps:
            return results_out

        groups = self.group_steps(steps)
        for g in groups:
            start = g[0]["index"]
            end = g[-1]["index"]
            label = f"{start}_{end}"
            step_details = "\n".join(s["step_text"] for s in g)
            query = self.build_query_for_group(g)

            # call pipeline
            matches_raw = self.pipeline.search(query, top_k=5, min_score=0.0, context=context)

            # format results
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

            results_out.append({
                "step_label": label,
                "step_details": step_details,
                "query": query,
                "matches": matches
            })

        return results_out


# -------------------------
# Adapter pipeline exposing stable API
# -------------------------
class OptimizedSearchPipeline:
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        # engine is the core searcher
        self.engine = MultiLevelSearchEngine(self.config)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0, context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        return self.engine.optimized_search(query, top_k, min_score, context)

    def clear_cache(self):
        self.engine.embedding_cache.clear()
        self.engine.result_cache.clear()
        # clear LRU cache wrapper too
        try:
            self.engine._cached_embedding.cache_clear()
        except Exception:
            pass


# -------------------------
# Instantiate global pipeline & processor (for core.py)
# -------------------------
_global_pipeline = OptimizedSearchPipeline()
gherkin_processor = GherkinSearchProcessor(_global_pipeline)


# -------------------------
# If run as script, simple debug demo
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_steps = [
        "Given the user is on the login page",
        "When the user enters username \"alice\"",
        "And the user enters password \"pwd123\"",
        "And clicks the Login button",
        "Then the user should see the dashboard"
    ]

    logger.info("Running local demo for optimized_search pipeline (this requires OpenSearch + Azure credentials)")

    # quick test — will call azure + opensearch
    res = gherkin_processor.analyze(demo_steps, context={"current_page": "LoginPage"})
    import json
    print(json.dumps(res, indent=2))
