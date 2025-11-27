"""
madl_engine.optimized_search (Azure OpenAI + OpenSearch integration)

- Embeddings: Azure OpenAI (via openai client)
- Vector DB: OpenSearch (knn_vector)
- All pipeline logic (normalization, query expansion, fusion, rerank, fuzzy, context) preserved.
"""

from __future__ import annotations
import os
import re
import time
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

# Optional features
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

# Azure OpenAI embedding client (OpenAI or Azure variant)
try:
    # prefer Azure-specific client if available in your environment
    from openai import AzureOpenAI as _AzureClient  # type: ignore
except Exception:
    try:
        from openai import OpenAI as _AzureClient  # fallback
    except Exception:
        _AzureClient = None  # will handle below

# OpenSearch client
try:
    from opensearchpy import OpenSearch
except Exception:
    OpenSearch = None

logger = logging.getLogger("madl_optimized_search")
logger.setLevel(logging.INFO)


# -------------------------
# Configuration dataclass
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
    index_name: str = os.getenv("OPENSEARCH_INDEX", "madl_methods_v2")

    # embedding dimension (default 1536 for ada-style)
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))


# -------------------------
# Lightweight wrapper for search results
# -------------------------
@dataclass
class SearchResult:
    score: float
    payload: Dict[str, Any]


# -------------------------
# Query Normalizer + Expander (kept identical to original logic)
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

    SYNONYM_REVERSE = {}
    for k, vals in SYNONYMS.items():
        for v in vals:
            SYNONYM_REVERSE[v] = k

    @classmethod
    def normalize(cls, text: str, is_gherkin: bool = False) -> str:
        if not text:
            return ""
        text = text.lower().strip()

        text = text.replace('_', ' ').replace('::', ' ').replace('->', ' ').replace('.', ' ')

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
# Keyword Booster (unchanged)
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
# Gherkin parser (kept)
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
# Context-aware scorer (kept)
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

        previous_steps = context.get('previous_steps', []) or []
        for prev in previous_steps:
            for page_class, keys in self.page_keywords.items():
                if any(k in prev.lower() for k in keys):
                    if page_class.lower() in method.get('class_name', '').lower():
                        score += 0.1
                        break

        return min(score, 1.0)


# -------------------------
# Azure OpenAI + OpenSearch integration
# -------------------------
# Env / defaults
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "madl_methods_v2")
VECTOR_FIELD = os.getenv("OPENSEARCH_VECTOR_FIELD", "embedding")

# Create OpenSearch client (best-effort; user environment likely has opensearchpy installed)
_client_os = None
if OpenSearch is not None:
    try:
        _client_os = OpenSearch(
            hosts=[OPENSEARCH_URL],
            use_ssl=True,
            verify_certs=False  # mirror your environment; change if you want cert verification
        )
        logger.info("OpenSearch client created.")
    except Exception as e:
        logger.warning(f"OpenSearch client init failed: {e}")
        _client_os = None
else:
    logger.warning("opensearchpy not installed; vector search will fail.")


# Azure OpenAI client factory
_azure_client = None
if _AzureClient is not None:
    try:
        _azure_client = _AzureClient(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_URL"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        )
        logger.info("Azure OpenAI client initialized.")
    except Exception as e:
        logger.warning(f"AzureOpenAI init failed: {e}")
        _azure_client = None
else:
    logger.warning("openai client library not available.")


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector using Azure OpenAI (or throw informative errors).
    Returns list[float].
    """
    if not _azure_client:
        raise RuntimeError("Azure OpenAI client is not initialized. Set AZURE_OPENAI_ENDPOINT & AZURE_OPENAI_API_KEY.")
    # Use the Azure client API to create embeddings
    resp = _azure_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=text
    )
    # response.data[0].embedding
    emb = resp.data[0].embedding
    return emb


# -------------------------
# Multi-level search engine (unchanged logic, OpenSearch vector calls plugged in)
# -------------------------
class MultiLevelSearchEngine:
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.normalizer = QueryNormalizer()
        self.booster = KeywordBooster()
        self.context_scorer = ContextAwareScorer()
        self.os_client = _client_os
        self.index_name = self.config.index_name or OPENSEARCH_INDEX

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
    def _cached_embedding(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        emb = generate_embedding(text)
        if self.config.cache_embeddings:
            self.embedding_cache[text] = emb
        return emb

    def _get_embedding(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        # call cached wrapper (lru_cache keyed by text)
        return self._cached_embedding(text)

    # ---- OpenSearch KNN vector search ----
    def _os_knn_search(self, vector: List[float], k: int) -> List[SearchResult]:
        if not self.os_client:
            raise RuntimeError("OpenSearch client not initialized.")
        body = {
            "size": k,
            "query": {
                "knn": {
                    VECTOR_FIELD: {
                        "vector": vector,
                        "k": k
                    }
                }
            }
        }
        resp = self.os_client.search(index=self.index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        results: List[SearchResult] = []
        for h in hits:
            score = float(h.get("_score", 0.0))
            payload = h.get("_source", {})
            results.append(SearchResult(score=score, payload=payload))
        return results

    # ---- OpenSearch text fallback (BM25 match) ----
    def _os_text_search(self, query: str, k: int) -> List[SearchResult]:
        if not self.os_client:
            raise RuntimeError("OpenSearch client not initialized.")
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
        resp = self.os_client.search(index=self.index_name, body=body)
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
            r.score = (r.score - min_s) / (max_s - min_s)
        return results

    # ---- fuse vector + text results (by doc id) ----
    def _fuse_results(self, vector_results: List[SearchResult], text_results: List[SearchResult]) -> List[SearchResult]:
        v_norm = self._normalize_scores(vector_results)
        t_norm = self._normalize_scores(text_results)

        merged: Dict[str, SearchResult] = {}

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
                r.score = 0.6 * r.score + 0.4 * float(ce_scores[i])
            results.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed: {e}")
        return results

    # ---- deduplicate by doc id preserving top score ----
    def _dedupe(self, results: List[SearchResult]) -> List[SearchResult]:
        seen = {}
        for r in results:
            did = f"{r.payload.get('class_name','')}.{r.payload.get('method_name','')}"
            if did not in seen:
                seen[did] = r
            else:
                if r.score > seen[did].score:
                    seen[did] = r
        final = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return final

    # ---- Search entry points: vector + text + expansion ----
    def level1_madl_search(self, query: str, top_k: int, min_score: float, filters=None) -> List[SearchResult]:
        """
        Level-1: vector KNN search (OpenSearch) with Azure embedding
        """
        # normalize / boost
        q_norm = self.normalizer.normalize(query) if self.config.use_normalization else query
        if self.config.use_keyword_boosting:
            q_norm = self.booster.boost_query(q_norm)

        # embed
        q_vec = self._get_embedding(q_norm)

        # KNN search
        vec_hits = self._os_knn_search(q_vec, k=max(top_k, 10))
        return vec_hits

    def level1_text_search(self, query: str, top_k: int) -> List[SearchResult]:
        return self._os_text_search(query, k=top_k)

    def search_with_query_expansion(self, query: str, top_k: int, min_score: float, filters=None) -> List[SearchResult]:
        variants = self.normalizer.expand_query(query)
        results_map: Dict[str, SearchResult] = {}
        for v in variants:
            hits = self.level1_text_search(v, top_k=self.config.text_fallback_k)
            for h in hits:
                did = f"{h.payload.get('class_name','')}.{h.payload.get('method_name','')}"
                if did not in results_map or h.score > results_map[did].score:
                    results_map[did] = h
        merged = list(results_map.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    # ---- public search API (keeps original pipeline) ----
    def optimized_search(self, query: str, top_k: int = 5, min_score: float = 0.0, context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        cache_key_raw = f"{query}|{top_k}|{min_score}|{context}"
        cache_hash = hashlib.md5(cache_key_raw.encode()).hexdigest()
        if self.config.cache_results and cache_hash in self.result_cache:
            results_cached, ts = self.result_cache[cache_hash]
            if time.time() - ts < self.config.cache_ttl:
                return results_cached

        q_norm = self.normalizer.normalize(query, is_gherkin=False) if self.config.use_normalization else query
        q_boosted = self.booster.boost_query(q_norm) if self.config.use_keyword_boosting else q_norm

        # embeddings + vector search
        vector_hits = []
        try:
            vector_hits = self.level1_madl_search(q_boosted, top_k=max(top_k * 3, self.config.text_fallback_k), min_score=min_score)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            vector_hits = []

        # text fallback / expansion
        text_hits = []
        if self.config.use_text_fallback:
            if self.config.use_query_expansion:
                text_hits = self.search_with_query_expansion(q_boosted, top_k=self.config.text_fallback_k, min_score=min_score)
            else:
                text_hits = self.level1_text_search(q_boosted, top_k=self.config.text_fallback_k)

        # fuse
        if vector_hits and text_hits:
            fused = self._fuse_results(vector_hits, text_hits)
        elif vector_hits:
            fused = self._normalize_scores(vector_hits)
        else:
            fused = self._normalize_scores(text_hits)

        # context boost
        if self.config.use_context_aware and context:
            fused = self._apply_context(fused, context)

        # rerank
        if self.config.use_reranking and self.reranker:
            fused_top = fused[: self.config.rerank_top_k]
            fused_top = self._rerank_with_cross_encoder(q_boosted, fused_top)
            fused = fused_top + fused[self.config.rerank_top_k :]

        # fuzzy
        if self.config.use_fuzzy_matching:
            fused = self._apply_fuzzy(q_boosted, fused)

        # dedupe
        fused = self._dedupe(fused)

        # filter
        final_filtered = [r for r in fused if r.score >= min_score] if min_score > 0 else fused

        final_filtered.sort(key=lambda x: x.score, reverse=True)
        final = final_filtered[:top_k]

        if self.config.cache_results:
            self.result_cache[cache_hash] = (final, time.time())

        return final


# -------------------------
# Adapter pipeline exposing stable API
# -------------------------
class OptimizedSearchPipeline:
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.engine = MultiLevelSearchEngine(self.config)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0, context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        return self.engine.optimized_search(query, top_k, min_score, context)

    def clear_cache(self):
        self.engine.embedding_cache.clear()
        self.engine.result_cache.clear()
        try:
            self.engine._cached_embedding.cache_clear()
        except Exception:
            pass


# -------------------------
# GherkinSearchProcessor wrapper (kept)
# -------------------------
class GherkinSearchProcessor:
    def __init__(self, pipeline: Optional[OptimizedSearchPipeline] = None):
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

            matches_raw = self.pipeline.search(query, top_k=5, min_score=0.0, context=context)

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
# Instantiate global pipeline & processor (for core.py)
# -------------------------
_global_pipeline = OptimizedSearchPipeline()
gherkin_processor = GherkinSearchProcessor(_global_pipeline)


# -------------------------
# If run as script, quick demo (requires env configured)
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

    logger.info("Running local demo for optimized_search pipeline")
    try:
        res = gherkin_processor.analyze(demo_steps, context={"current_page": "LoginPage"})
        import json
        print(json.dumps(res, indent=2))
    except Exception as e:
        logger.exception("Demo failed: %s", e)
