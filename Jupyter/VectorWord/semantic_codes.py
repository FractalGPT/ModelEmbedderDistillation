"""Семантические коды для BM25 в RAG.

Модуль собирает весь конвейер из документа «Семантические коды для BM25 в RAG»:

1. Разбиение текста на предложения и слова.
2. Векторы на каждый токен (ColBERT-голова BGE-M3) и усреднение токенов внутри слова.
3. Бинаризация: PCA (+ поворот ITQ) или автоэнкодер с бинарным слоем.
4. Запись кода как токена вида §h0_0110100111§, несколько хешей на слово.
5. Точные сущности (телефоны, email, даты, суммы, номера, NER) без изменений.
6. BM25 (свой, на разреженных матрицах), многополевой BM25 с весами и RRF.
7. Базовые методы: BM25 по словам, dense BGE-M3, ColBERT MaxSim.
8. Метрики Recall@k, nDCG@10, MRR, размер индекса, скорость поиска.

Ноутбук semantic_codes.ipynb только вызывает функции отсюда.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Текст: предложения, слова, леммы
# ---------------------------------------------------------------------------

# Комбинируемые диакритики (U+0300–U+036F) и мягкий перенос (U+00AD) считаются частью слова:
# в Википедии ударения пишутся как «Бреча́лов», и без этого слово режется на «Бреча» + «лов».
_W = r"[\ẁ-ͯ­]"
WORD_RE = re.compile(rf"{_W}+(?:[-']{_W}+)*", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"«(]?[A-ZА-ЯЁ0-9])")
_STRIP_RE = re.compile(r"[̀́­]")  # ударения и мягкий перенос


def normalize_text(text: str) -> str:
    """Убирает знаки ударения и мягкие переносы, приводит к NFC (й, ё остаются)."""
    return unicodedata.normalize("NFC", _STRIP_RE.sub("", unicodedata.normalize("NFC", text)))


def split_sentences(text: str) -> List[str]:
    """Делит текст на предложения: razdel, если установлен, иначе регулярка."""
    text = text.strip()
    if not text:
        return []
    try:
        from razdel import sentenize

        return [s.text for s in sentenize(text) if s.text.strip()]
    except ImportError:
        return [s for s in _SENT_RE.split(text) if s.strip()]


def word_spans(text: str) -> List[Tuple[int, int]]:
    """Границы слов (начало, конец) в символах."""
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def simple_tokenize(text: str) -> List[str]:
    """Слова в нижнем регистре, без ударений, ё → е."""
    return [normalize_text(w).lower().replace("ё", "е") for w in WORD_RE.findall(text)]


_MORPH = None


def _get_morph():
    global _MORPH
    if _MORPH is None:
        try:
            import pymorphy3 as pm
        except ImportError:
            try:
                import pymorphy2 as pm
            except ImportError:
                pm = None
        _MORPH = pm.MorphAnalyzer() if pm is not None else False
    return _MORPH


@lru_cache(maxsize=500_000)
def lemmatize_word(word: str) -> str:
    """Лемма через pymorphy3/pymorphy2; без них слово возвращается как есть."""
    morph = _get_morph()
    if not morph or not re.search(r"[а-яё]", word):
        return word
    return morph.parse(word)[0].normal_form.replace("ё", "е")


def lemmatize(text: str) -> List[str]:
    """Поток лемм для поля «исходные слова»."""
    return [lemmatize_word(w) for w in simple_tokenize(text)]


# ---------------------------------------------------------------------------
# Точные сущности
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(
    r"(?<![\d\w])(?:\+7|8|7)?[\s\-]*\(?\d{3}\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
)
_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_DATE_NUM_RE = re.compile(r"(?<!\d)(\d{1,2})[./\-](\d{1,2})[./\-](\d{4}|\d{2})(?!\d)")
_DATE_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_DATE_TXT_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[яй]|июн\w*|июл\w*|"
    r"август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(\d{4})",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(\d[\d\s]*(?:[.,]\d+)?)\s*(тыс\.?|млн\.?|млрд\.?)?\s*"
    r"(руб(?:лей|ля|ль)?\.?|₽|\$|usd|долл\w*|eur|€|евро)",
    re.IGNORECASE,
)
_DOCNUM_RE = re.compile(r"№\s*([\w\-/]+)")
_LONGNUM_RE = re.compile(r"(?<![\d\w])\d{4,}(?![\d\w])")

_CURRENCY = {"₽": "RUB", "$": "USD", "€": "EUR"}


def _norm_currency(cur: str) -> str:
    cur = cur.lower()
    if cur in _CURRENCY:
        return _CURRENCY[cur]
    if cur.startswith("руб"):
        return "RUB"
    if cur.startswith("долл") or cur == "usd":
        return "USD"
    return "EUR"


def extract_regex_entities(text: str) -> List[str]:
    """Телефоны, email, даты, суммы, номера документов в едином виде.

    Возвращает токены с префиксом типа: ent:phone:+79161234567, ent:date:2023-03-12 ...
    """
    ents: List[str] = []
    used = []  # занятые интервалы, чтобы телефон не стал ещё и «длинным числом»

    def add(kind: str, value: str, span: Tuple[int, int]):
        ents.append(f"ent:{kind}:{value}")
        used.append(span)

    for m in _EMAIL_RE.finditer(text):
        add("email", m.group(0).lower(), m.span())
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 10:
            continue
        add("phone", "+7" + digits[-10:], m.span())
    for m in _DATE_NUM_RE.finditer(text):
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        y = int(y) + (2000 if len(y) == 2 and int(y) < 50 else 1900 if len(y) == 2 else 0)
        if 1 <= d <= 31 and 1 <= mo <= 12:
            add("date", f"{y:04d}-{mo:02d}-{d:02d}", m.span())
    for m in _DATE_ISO_RE.finditer(text):
        add("date", m.group(0), m.span())
    for m in _DATE_TXT_RE.finditer(text):
        word = m.group(2).lower()
        mo = next(v for k, v in _MONTHS.items() if word.startswith(k))
        add("date", f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}", m.span())
    for m in _MONEY_RE.finditer(text):
        num = float(re.sub(r"\s", "", m.group(1)).replace(",", "."))
        mult = (m.group(2) or "").lower()
        num *= 1e3 if mult.startswith("тыс") else 1e6 if mult.startswith("млн") else 1e9 if mult.startswith("млрд") else 1
        add("money", f"{num:g}{_norm_currency(m.group(3))}", m.span())
    for m in _DOCNUM_RE.finditer(text):
        add("num", m.group(1).lower(), m.span())
    for m in _LONGNUM_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in used):
            add("num", m.group(0), m.span())
    return ents


_NATASHA = None


def _get_natasha():
    global _NATASHA
    if _NATASHA is None:
        try:
            from natasha import Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter

            emb = NewsEmbedding()
            _NATASHA = (Doc, Segmenter(), NewsNERTagger(emb), MorphVocab())
        except ImportError:
            _NATASHA = False
    return _NATASHA


def extract_ner_entities(text: str) -> List[str]:
    """Имена, организации, топонимы через natasha (если установлена).

    Возвращает токены вида ent:per:иван_петров. Без natasha — пустой список.
    """
    nat = _get_natasha()
    if not nat:
        return []
    Doc, segmenter, tagger, morph_vocab = nat
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(tagger)
    out = []
    for span in doc.spans:
        if re.search(r"[\d@]", span.text):  # email/номера уже взяты регулярками
            continue
        value = "_".join(lemmatize_word(w) for w in simple_tokenize(span.text))
        if value:
            out.append(f"ent:{span.type.lower()}:{value}")
    return out


def extract_entities(text: str, use_ner: bool = True) -> List[str]:
    """Все точные сущности текста: регулярки + NER."""
    ents = extract_regex_entities(text)
    if use_ner:
        ents += extract_ner_entities(text)
    return ents


# ---------------------------------------------------------------------------
# Модель: векторы на каждый токен и на каждое слово
# ---------------------------------------------------------------------------


class TokenEncoder:
    """Обёртка над BGE-M3 (или любой HF-моделью) с векторами на каждый токен.

    Для BGE-M3 подгружается ColBERT-голова (colbert_linear.pt) и берётся её выход,
    как в FlagEmbedding: colbert = normalize(Linear(hidden[:, 1:])).
    Dense-вектор: нормированный CLS последнего слоя.
    Для других моделей (или use_colbert_head=False) берутся сырые скрытые состояния.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        max_length: int = 512,
        fp16: bool = True,
        use_colbert_head: bool = True,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)
        if fp16 and self.device.startswith("cuda"):
            self.model.half()
        self.colbert = None
        if use_colbert_head:
            self.colbert = self._load_colbert_head(model_name)
        self.dim = self.colbert.out_features if self.colbert is not None else self.model.config.hidden_size

    def _load_colbert_head(self, model_name: str):
        torch = self.torch
        path = os.path.join(model_name, "colbert_linear.pt")
        if not os.path.exists(path):
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(model_name, "colbert_linear.pt")
            except Exception as e:  # у модели нет ColBERT-головы
                print(f"[TokenEncoder] ColBERT-голова не найдена ({type(e).__name__}), берутся скрытые состояния")
                return None
        state = torch.load(path, map_location="cpu", weights_only=True)
        out_f, in_f = state["weight"].shape
        head = torch.nn.Linear(in_f, out_f)
        head.load_state_dict(state)
        head = head.eval().to(self.device)
        if next(self.model.parameters()).dtype == torch.float16:
            head.half()
        return head

    def encode_batch(self, texts: Sequence[str]):
        """Один батч → (dense [B, d], список токен-векторов, список offsets).

        Токен-векторы без CLS/SEP/паддинга, нормированные, float16 numpy.
        offsets — границы токенов в символах исходного текста.
        """
        torch = self.torch
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").numpy()
        special = enc.pop("special_tokens_mask").numpy().astype(bool)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self.model(**enc).last_hidden_state
            dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
            tok = self.colbert(hidden) if self.colbert is not None else hidden
            tok = torch.nn.functional.normalize(tok.float(), dim=-1)
        dense = dense.cpu().numpy().astype(np.float16)
        tok = tok.cpu().numpy().astype(np.float16)
        mask = enc["attention_mask"].cpu().numpy().astype(bool) & ~special
        tok_list, off_list = [], []
        for i in range(len(texts)):
            tok_list.append(tok[i][mask[i]])
            off_list.append(offsets[i][mask[i]])
        return dense, tok_list, off_list


def pool_words(text: str, tok_vecs: np.ndarray, offsets: np.ndarray):
    """Усредняет векторы токенов внутри каждого слова.

    Токен относится к слову, если их интервалы символов пересекаются.
    Возвращает (слова, векторы слов [n_words, d] float16).
    """
    words, vecs = [], []
    for s, e in word_spans(text):
        idx = np.where((offsets[:, 0] < e) & (offsets[:, 1] > s) & (offsets[:, 1] > offsets[:, 0]))[0]
        if len(idx) == 0:  # слово обрезано по max_length
            continue
        v = tok_vecs[idx].astype(np.float32).mean(0)
        v /= np.linalg.norm(v) + 1e-8
        words.append(text[s:e])
        vecs.append(v)
    if not vecs:
        return [], np.zeros((0, tok_vecs.shape[1] if tok_vecs.ndim == 2 else 1), np.float16)
    return words, np.stack(vecs).astype(np.float16)


@dataclass
class EncodedCorpus:
    """Закодированные тексты.

    dense       [N, d]   — один вектор на текст (BGE-M3 dense).
    tok_vecs    list[N]  — векторы токенов всего текста (для ColBERT MaxSim).
    words       list[N]  — слова каждого текста (по предложениям).
    word_vecs   [W, d]   — векторы всех слов подряд, word_offsets[N+1] — границы текстов.
    """

    dense: np.ndarray
    tok_vecs: List[np.ndarray]
    words: List[List[str]]
    word_vecs: np.ndarray
    word_offsets: np.ndarray
    word_sent: Optional[np.ndarray] = None  # [W] номер предложения каждого слова (сквозной)

    def doc_word_vecs(self, i: int) -> np.ndarray:
        return self.word_vecs[self.word_offsets[i]: self.word_offsets[i + 1]]

    def __len__(self):
        return len(self.words)


def _batched_by_length(texts: Sequence[str], batch_size: int):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), batch_size):
        yield order[start: start + batch_size]


def encode_corpus(
    encoder: TokenEncoder,
    texts: Sequence[str],
    batch_size: int = 32,
    sentence_level_words: bool = True,
    keep_tokens: bool = True,
    verbose: bool = True,
) -> EncodedCorpus:
    """Кодирует тексты.

    Проход 1 (весь текст): dense-вектор и векторы токенов для ColBERT.
    Проход 2 (по предложениям, если sentence_level_words): векторы слов в контексте предложения.
    При sentence_level_words=False векторы слов берутся из прохода 1.
    """
    n = len(texts)
    dense = np.zeros((n, encoder.dim), np.float16)
    tok_vecs: List[Optional[np.ndarray]] = [None] * n
    doc_words: List[List[str]] = [[] for _ in range(n)]
    doc_wvecs: List[List[np.ndarray]] = [[] for _ in range(n)]
    doc_wsent: List[List[int]] = [[] for _ in range(n)]
    t0 = time.perf_counter()

    for bi, idx in enumerate(_batched_by_length(texts, batch_size)):
        d, toks, offs = encoder.encode_batch([texts[i] for i in idx])
        for j, i in enumerate(idx):
            dense[i] = d[j]
            if keep_tokens:
                tok_vecs[i] = toks[j]
            if not sentence_level_words:
                w, v = pool_words(texts[i], toks[j], offs[j])
                doc_words[i], doc_wvecs[i], doc_wsent[i] = w, [v], [i] * len(w)
        if verbose and bi % 20 == 0:
            print(f"\r[тексты] {min((bi + 1) * batch_size, n)}/{n}  {time.perf_counter() - t0:.0f}s", end="")
    if verbose:
        print()

    if sentence_level_words:
        sents, owner = [], []
        for i, t in enumerate(texts):
            for s in split_sentences(t):
                sents.append(s)
                owner.append(i)
        sent_words: List = [None] * len(sents)
        for bi, idx in enumerate(_batched_by_length(sents, batch_size)):
            _, toks, offs = encoder.encode_batch([sents[i] for i in idx])
            for j, i in enumerate(idx):
                sent_words[i] = pool_words(sents[i], toks[j], offs[j])
            if verbose and bi % 20 == 0:
                print(f"\r[предложения] {min((bi + 1) * batch_size, len(sents))}/{len(sents)}  "
                      f"{time.perf_counter() - t0:.0f}s", end="")
        if verbose:
            print()
        for k, ((w, v), i) in enumerate(zip(sent_words, owner)):  # порядок предложений сохраняется
            doc_words[i].extend(w)
            doc_wvecs[i].append(v)
            doc_wsent[i].extend([k] * len(w))

    lens = [len(w) for w in doc_words]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    parts = [np.concatenate(v) if v else np.zeros((0, encoder.dim), np.float16) for v in doc_wvecs]
    parts = [p if len(p) else np.zeros((0, encoder.dim), np.float16) for p in parts]
    word_vecs = np.concatenate(parts).astype(np.float16) if offsets[-1] else np.zeros((0, encoder.dim), np.float16)
    word_sent = np.array([k for s in doc_wsent for k in s], np.int64)
    return EncodedCorpus(dense, tok_vecs if keep_tokens else [], doc_words, word_vecs, offsets, word_sent)


CACHE_VERSION = 3  # увеличивать при изменении предобработки текста, чтобы не подхватить старый кеш


def encode_cached(encoder_factory: Callable[[], TokenEncoder], texts: Sequence[str], path: str, **kw) -> EncodedCorpus:
    """encode_corpus с кешем на диске (pickle). encoder_factory вызывается только без кеша."""
    root, ext = os.path.splitext(path)
    path = f"{root}_v{CACHE_VERSION}{ext}"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    enc = encode_corpus(encoder_factory(), texts, **kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"  # запись через временный файл: при сбое не останется битого кеша
    with open(tmp, "wb") as f:
        pickle.dump(enc, f, protocol=4)
    os.replace(tmp, path)
    return enc


# ---------------------------------------------------------------------------
# Бинаризация: PCA + ITQ
# ---------------------------------------------------------------------------


def _sample_rows(X: np.ndarray, max_rows: Optional[int], seed: int) -> np.ndarray:
    if max_rows is None or len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), max_rows, replace=False)]


class PCAITQ:
    """PCA до n_bits компонент, затем поворот ITQ и знак.

    use_itq=False — чистый PCA (для сравнения), random_rotation=True — случайный поворот без ITQ.
    ITQ чередует: B = sign(V R); C = Bᵀ V = U S Wᵀ; R = W Uᵀ.
    """

    def __init__(self, n_bits: int = 10, n_iter: int = 50, use_itq: bool = True,
                 random_rotation: bool = False, seed: int = 0, max_fit_rows: Optional[int] = 200_000,
                 skip_top: int = 0):
        self.n_bits, self.n_iter, self.use_itq, self.skip_top = n_bits, n_iter, use_itq, skip_top
        self.random_rotation, self.seed, self.max_fit_rows = random_rotation, seed, max_fit_rows
        self.quant_loss: List[float] = []

    def fit(self, X: np.ndarray) -> "PCAITQ":
        X = _sample_rows(X, self.max_fit_rows, self.seed).astype(np.float32)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / len(Xc)
        eigval, eigvec = np.linalg.eigh(cov)
        top = np.argsort(eigval)[::-1][self.skip_top: self.skip_top + self.n_bits]
        self.components_ = eigvec[:, top]  # [d, n_bits]
        self.explained_ = eigval[top] / eigval.sum()
        V = Xc @ self.components_
        rng = np.random.default_rng(self.seed)
        R = np.linalg.qr(rng.standard_normal((self.n_bits, self.n_bits)))[0]
        if not (self.use_itq or self.random_rotation):
            R = np.eye(self.n_bits)
        if self.use_itq:
            for _ in range(self.n_iter):
                B = np.sign(V @ R)
                B[B == 0] = 1
                self.quant_loss.append(float(((B - V @ R) ** 2).mean()))
                U, _, Wt = np.linalg.svd(B.T @ V)
                R = Wt.T @ U.T
        self.R_ = R.astype(np.float32)
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        """Непрерывная проекция до знака [n, n_bits]."""
        return (X.astype(np.float32) - self.mean_) @ self.components_ @ self.R_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Бинарные коды [n, n_bits] uint8."""
        return (self.project(X) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Бинаризация: автоэнкодер
# ---------------------------------------------------------------------------


class AEBinarizer:
    """Автоэнкодер с бинарным слоем посередине.

    Прямой проход: p = sigmoid(logits + шум), b = round(p) (straight-through: градиент через p).
    Потери: реконструкция (1 - cos), баланс бит (mean(p) → 0.5), декорреляция бит.
    """

    def __init__(self, n_bits: int = 10, hidden: int = 512, noise: float = 1.0,
                 w_balance: float = 1.0, w_decorr: float = 1.0, epochs: int = 20,
                 batch_size: int = 1024, lr: float = 1e-3, device: Optional[str] = None,
                 seed: int = 0, max_fit_rows: Optional[int] = 500_000):
        self.n_bits, self.hidden, self.noise = n_bits, hidden, noise
        self.w_balance, self.w_decorr = w_balance, w_decorr
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.device, self.seed, self.max_fit_rows = device, seed, max_fit_rows
        self.history: List[Dict[str, float]] = []

    def _build(self, d: int):
        import torch.nn as nn

        self.enc = nn.Sequential(nn.Linear(d, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.n_bits))
        self.dec = nn.Sequential(nn.Linear(self.n_bits, self.hidden), nn.GELU(), nn.Linear(self.hidden, d))

    def fit(self, X: np.ndarray, verbose: bool = True) -> "AEBinarizer":
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        X = _sample_rows(X, self.max_fit_rows, self.seed).astype(np.float32)
        self.mean_ = X.mean(0)
        Xt = torch.from_numpy(X - self.mean_).to(device)
        self._build(X.shape[1])
        self.enc.to(device), self.dec.to(device)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr)
        eye = torch.eye(self.n_bits, device=device, dtype=torch.bool)
        for ep in range(self.epochs):
            perm = torch.randperm(len(Xt), device=device)
            sums = Counter()
            nb = 0
            for s in range(0, len(Xt), self.batch_size):
                x = Xt[perm[s: s + self.batch_size]]
                logits = self.enc(x)
                p = torch.sigmoid(logits + self.noise * torch.randn_like(logits))
                b = (p > 0.5).float() + p - p.detach()  # straight-through
                x_hat = self.dec(b)
                rec = (1 - F.cosine_similarity(x_hat, x, dim=-1)).mean()
                bal = ((p.mean(0) - 0.5) ** 2).sum()
                z = 2 * p - 1
                z = z - z.mean(0)
                cov = z.T @ z / len(z)
                dec = (cov[~eye] ** 2).mean()
                loss = rec + self.w_balance * bal + self.w_decorr * dec
                opt.zero_grad()
                loss.backward()
                opt.step()
                sums.update(loss=loss.item(), rec=rec.item(), balance=bal.item(), decorr=dec.item())
                nb += 1
            row = {k: v / nb for k, v in sums.items()}
            row["epoch"] = ep
            self.history.append(row)
            if verbose:
                print(f"\r[AE] epoch {ep + 1}/{self.epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"), end="")
        if verbose:
            print()
        self.enc.eval().cpu(), self.dec.eval().cpu()
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.enc(torch.from_numpy(X.astype(np.float32) - self.mean_)).numpy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self.project(X) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Преобразования векторов слов перед бинаризацией
# ---------------------------------------------------------------------------


def _renorm(V: np.ndarray) -> np.ndarray:
    V = V.astype(np.float32)
    return (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)).astype(np.float16)


def center_by_group(enc: EncodedCorpus, by: str = "sentence", alpha: float = 1.0) -> EncodedCorpus:
    """Вычитает из вектора слова средний вектор его предложения (by='sentence') или текста (by='doc').

    Убирает общую для всех слов составляющую контекста (тему), оставляя то, чем слово выделяется.
    alpha < 1 — частичное вычитание.
    """
    if by == "sentence" and enc.word_sent is None:
        raise ValueError("нет word_sent: перекодируйте корпус свежей версией encode_corpus")
    groups = enc.word_sent if by == "sentence" else np.repeat(np.arange(len(enc)), np.diff(enc.word_offsets))
    _, inv = np.unique(groups, return_inverse=True)
    V = enc.word_vecs.astype(np.float32)
    M = sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))))
    means = (M @ V) / np.asarray(M.sum(1))
    return replace(enc, word_vecs=_renorm(V - alpha * means[inv]))


def shift_domain(enc: EncodedCorpus, src_mean: np.ndarray, dst_mean: np.ndarray) -> EncodedCorpus:
    """Сдвигает облако векторов слов: v - src_mean + dst_mean.

    Для запросов: src_mean — среднее слов запросов, dst_mean — среднее слов документов.
    Убирает систематический сдвиг «короткий вопрос vs абзац».
    """
    V = enc.word_vecs.astype(np.float32) - src_mean + dst_mean
    return replace(enc, word_vecs=V.astype(np.float16))


def align_queries(doc_enc: EncodedCorpus, q_enc: EncodedCorpus) -> EncodedCorpus:
    """Переносит среднее векторов слов запросов в среднее слов документов."""
    return shift_domain(q_enc, q_enc.word_vecs.astype(np.float32).mean(0), doc_enc.word_vecs.astype(np.float32).mean(0))


def code_overlap(q_codes: Sequence[Sequence[str]], d_codes: Sequence[Sequence[str]], qrels,
                 n_random: int = 20, seed: int = 0) -> Dict[str, float]:
    """Доля токенов-кодов запроса, встречающихся в релевантном и в случайном документе.

    Чем больше разрыв relevant - random, тем лучше коды отделяют нужный документ.
    """
    rng = np.random.default_rng(seed)
    rel, rnd = [], []
    for qi, r in enumerate(qrels):
        q = set(q_codes[qi])
        if not q or not r:
            continue
        rel.extend(len(q & set(d_codes[d])) / len(q) for d in r)
        rnd.extend(len(q & set(d_codes[d])) / len(q) for d in rng.integers(0, len(d_codes), n_random))
    return {"overlap_relevant": float(np.mean(rel)), "overlap_random": float(np.mean(rnd)),
            "overlap_gap": float(np.mean(rel) - np.mean(rnd))}


_STOP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "PRED"}


@lru_cache(maxsize=500_000)
def is_content_word(word: str) -> bool:
    """Знаменательное слово: не предлог, союз, частица, междометие, местоимение."""
    morph = _get_morph()
    w = normalize_text(word).lower()
    if not morph:
        return len(w) > 2
    return morph.parse(w)[0].tag.POS not in _STOP_POS


def filter_words(enc: EncodedCorpus, keep: Callable[[str], bool] = is_content_word) -> EncodedCorpus:
    """Оставляет только слова, для которых keep(слово) истинно (по умолчанию — без стоп-слов)."""
    mask = np.array([keep(w) for ws in enc.words for w in ws], bool)
    words, counts = [], []
    for i, ws in enumerate(enc.words):
        m = mask[enc.word_offsets[i]: enc.word_offsets[i + 1]]
        words.append([w for w, k in zip(ws, m) if k])
        counts.append(int(m.sum()))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sent = enc.word_sent[mask] if enc.word_sent is not None else None
    return replace(enc, words=words, word_vecs=enc.word_vecs[mask], word_offsets=offsets, word_sent=sent)


# ---------------------------------------------------------------------------
# Коды → токены, несколько хешей, статистика кодов
# ---------------------------------------------------------------------------


def bits_to_str(bits: np.ndarray) -> np.ndarray:
    """[n, n_bits] uint8 → массив строк '0110100111'."""
    if len(bits) == 0:
        return np.array([], dtype=str)
    return np.array(["".join(map(str, row)) for row in bits])


def code_token(bits_str: str, hash_id: int = 0) -> str:
    """Код как токен с префиксом и суффиксом: §h0_0110100111§."""
    return f"§h{hash_id}_{bits_str}§"


def fit_hashes(X: np.ndarray, n_bits: int = 10, n_hashes: int = 1, method: str = "itq", **kw) -> list:
    """Обучает n_hashes независимых бинаризаторов (разные seed → разные повороты/подвыборки).

    method: 'itq' | 'pca' | 'random' (PCA + случайный поворот) | 'ae'.
    """
    if method == "pca" and n_hashes > 1:
        print("[fit_hashes] PCA без поворота детерминирован: все хеши одинаковы, n_hashes>1 ничего не даёт")
    out = []
    for h in range(n_hashes):
        if method == "ae":
            out.append(AEBinarizer(n_bits=n_bits, seed=h, **kw).fit(X))
        else:
            out.append(PCAITQ(n_bits=n_bits, seed=h, use_itq=method == "itq",
                              random_rotation=method == "random", **kw).fit(X))
    return out


def corpus_code_tokens(enc: EncodedCorpus, binarizers: list) -> List[List[str]]:
    """Тексты, переписанные на «семантическом словаре»: список токенов-кодов на каждый текст."""
    per_hash = []
    for h, binz in enumerate(binarizers):
        strs = bits_to_str(binz.transform(enc.word_vecs)) if len(enc.word_vecs) else np.array([])
        per_hash.append(strs)
    docs = []
    for i in range(len(enc)):
        a, b = enc.word_offsets[i], enc.word_offsets[i + 1]
        toks = []
        for k in range(a, b):  # хеши одного слова идут подряд
            toks.extend(code_token(per_hash[h][k], h) for h in range(len(binarizers)))
        docs.append(toks)
    return docs


def code_stats(binarizer, X: np.ndarray, eps: float = 0.05) -> Dict[str, float]:
    """Качество кодов: заполненность словаря, энтропия, баланс бит, корреляция, точки у нуля."""
    proj = binarizer.project(X)
    bits = (proj > 0).astype(np.uint8)
    n_bits = bits.shape[1]
    ids = bits @ (1 << np.arange(n_bits)[::-1])
    cnt = np.bincount(ids, minlength=2 ** n_bits)
    p = cnt[cnt > 0] / cnt.sum()
    z = 2.0 * bits - 1
    corr = np.corrcoef(z.T) if n_bits > 1 else np.ones((1, 1))
    off = corr[~np.eye(n_bits, dtype=bool)]
    scale = np.abs(proj).std(0, keepdims=True) + 1e-8
    return {
        "used_codes": int((cnt > 0).sum()),
        "used_share": float((cnt > 0).mean()),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "max_entropy": float(n_bits),
        "top1_code_share": float(cnt.max() / cnt.sum()),
        "bit_balance_dev": float(np.abs(bits.mean(0) - 0.5).mean()),
        "mean_abs_bit_corr": float(np.nan_to_num(np.abs(off)).mean()) if len(off) else 0.0,
        "near_zero_share": float((np.abs(proj) / scale < eps).mean()),
    }


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25:
    """BM25 на разреженной матрице: веса термов считаются заранее, поиск — одно умножение."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: Sequence[Sequence[str]]) -> "BM25":
        self.vocab: Dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        n_docs = len(docs)
        tf = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n_docs, max(len(self.vocab), 1)))
        dl = np.array([len(d) for d in docs], np.float32)
        avgdl = dl.mean() if n_docs and dl.mean() > 0 else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * dl / avgdl)
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sp.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.n_docs = n_docs
        return self

    def scores(self, query: Sequence[str]) -> np.ndarray:
        cnt = Counter(t for t in query if t in self.vocab)
        if not cnt:
            return np.zeros(self.n_docs, np.float32)
        ids = np.array([self.vocab[t] for t in cnt])
        q = np.array(list(cnt.values()), np.float32)
        return np.asarray(self.W[:, ids] @ q).ravel()

    def score_matrix(self, queries: Sequence[Sequence[str]]) -> np.ndarray:
        return np.stack([self.scores(q) for q in queries]) if queries else np.zeros((0, self.n_docs))

    def size_bytes(self) -> int:
        """Размер инвертированного индекса (данные + индексы матрицы) без словаря."""
        return int(self.W.data.nbytes + self.W.indices.nbytes + self.W.indptr.nbytes)

    def vocab_bytes(self) -> int:
        return int(sum(len(t.encode("utf-8")) + 4 for t in self.vocab))


class MultiFieldBM25:
    """Отдельный BM25 на каждое поле (коды, сущности, слова) и взвешенная сумма оценок."""

    def __init__(self, weights: Dict[str, float], k1: float = 1.2, b: float = 0.75,
                 params: Optional[Dict[str, Tuple[float, float]]] = None):
        self.weights = weights
        params = params or {}
        self.indexes = {f: BM25(*params.get(f, (k1, b))) for f in weights}

    def fit(self, fields: Dict[str, Sequence[Sequence[str]]]) -> "MultiFieldBM25":
        for f, idx in self.indexes.items():
            idx.fit(fields[f])
        return self

    def field_scores(self, queries: Dict[str, Sequence[Sequence[str]]]) -> Dict[str, np.ndarray]:
        return {f: idx.score_matrix(queries[f]) for f, idx in self.indexes.items()}

    def score_matrix(self, queries: Dict[str, Sequence[Sequence[str]]],
                     weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        weights = weights or self.weights
        fs = self.field_scores(queries)
        return sum(weights[f] * s for f, s in fs.items() if weights.get(f, 0))

    def size_bytes(self) -> int:
        return sum(i.size_bytes() for i in self.indexes.values())


def combine_scores(field_scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Взвешенная сумма заранее посчитанных матриц оценок (для быстрого перебора весов)."""
    return sum(w * field_scores[f] for f, w in weights.items() if w)


def rrf(score_mats: Sequence[np.ndarray], k: int = 60, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Reciprocal Rank Fusion нескольких матриц оценок [n_q, n_docs]."""
    weights = weights or [1.0] * len(score_mats)
    out = np.zeros_like(score_mats[0], dtype=np.float32)
    for w, s in zip(weights, score_mats):
        ranks = np.argsort(np.argsort(-s, axis=1), axis=1)
        out += w / (k + ranks + 1)
    return out


# ---------------------------------------------------------------------------
# Плотный поиск и ColBERT MaxSim
# ---------------------------------------------------------------------------


def dense_scores(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) @ d.astype(np.float32).T


def colbert_scores(q_toks: Sequence[np.ndarray], d_toks: Sequence[np.ndarray],
                   device: Optional[str] = None, doc_chunk: int = 512) -> np.ndarray:
    """MaxSim: для каждого токена запроса — максимум сходства с токенами документа, сумма."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    out = np.zeros((len(q_toks), len(d_toks)), np.float32)
    for s in range(0, len(d_toks), doc_chunk):
        chunk = d_toks[s: s + doc_chunk]
        L = max(1, max(len(x) for x in chunk))
        D = torch.zeros((len(chunk), L, chunk[0].shape[1]), dtype=dtype)
        M = torch.zeros((len(chunk), L), dtype=torch.bool)
        for j, x in enumerate(chunk):
            D[j, : len(x)] = torch.from_numpy(x.astype(np.float32))
            M[j, : len(x)] = True
        D, M = D.to(device), M.to(device)
        for qi, q in enumerate(q_toks):
            if len(q) == 0:
                continue
            Q = torch.from_numpy(q.astype(np.float32)).to(device, dtype)
            sim = torch.einsum("qd,nld->nql", Q, D).masked_fill(~M[:, None, :], -1e4)
            out[qi, s: s + len(chunk)] = sim.max(-1).values.float().sum(-1).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------


def evaluate(scores: np.ndarray, qrels: Sequence[Dict[int, float]], ks: Sequence[int] = (1, 5, 10, 100),
             ndcg_k: int = 10) -> Dict[str, float]:
    """Recall@k, nDCG@10, MRR по матрице оценок [n_q, n_docs] и qrels (индекс документа → релевантность)."""
    top_n = max(max(ks), ndcg_k)
    top_n = min(top_n, scores.shape[1])
    top = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, 1), axis=1), 1)
    res = {f"Recall@{k}": 0.0 for k in ks}
    res[f"nDCG@{ndcg_k}"] = 0.0
    res["MRR"] = 0.0
    for qi, rel in enumerate(qrels):
        rel = {d: r for d, r in rel.items() if r > 0}
        if not rel:
            continue
        ranked = top[qi]
        hits = np.array([d in rel for d in ranked])
        for k in ks:
            res[f"Recall@{k}"] += hits[:k].sum() / len(rel)
        gains = np.array([rel.get(d, 0.0) for d in ranked[:ndcg_k]])
        dcg = ((2 ** gains - 1) / np.log2(np.arange(2, len(gains) + 2))).sum()
        ideal = np.sort(list(rel.values()))[::-1][:ndcg_k]
        idcg = ((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum()
        res[f"nDCG@{ndcg_k}"] += dcg / idcg if idcg else 0
        first = np.where(hits)[0]
        if len(first):
            res["MRR"] += 1 / (first[0] + 1)
    n = sum(1 for r in qrels if any(v > 0 for v in r.values()))
    return {k: v / max(n, 1) for k, v in res.items()}


def timed(fn: Callable, *args, repeat: int = 1, **kw):
    """Выполняет fn и возвращает (результат, среднее время в секундах)."""
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args, **kw)
    return out, (time.perf_counter() - t0) / repeat


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


@dataclass
class RetrievalData:
    doc_ids: List[str]
    docs: List[str]
    query_ids: List[str]
    queries: List[str]
    qrels: List[Dict[int, float]]  # по индексу запроса: индекс документа → релевантность
    name: str = ""

    def __repr__(self):
        n_rel = sum(len(q) for q in self.qrels)
        return f"RetrievalData({self.name}: {len(self.docs)} docs, {len(self.queries)} queries, {n_rel} qrels)"


def _pick(cols: Sequence[str], *names: str) -> str:
    for n in names:
        if n in cols:
            return n
    raise KeyError(f"нет ни одной из колонок {names} среди {list(cols)}")


def _first_split(ds):
    return ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds


def load_mteb_retrieval(name: str = "ai-forever/rubq-retrieval", n_queries: Optional[int] = 300,
                        n_distractors: Optional[int] = 2000, seed: int = 0, qrels_split: str = "test",
                        with_title: bool = True) -> RetrievalData:
    """Загружает ретривал-датасет в формате MTEB/BEIR (конфиги corpus, queries, default).

    Подходит для ruMTEB: ai-forever/rubq-retrieval, ai-forever/ria-news-retrieval и др.
    Для скорости берётся n_queries запросов, их релевантные документы и n_distractors случайных.
    """
    from datasets import load_dataset

    corpus = _first_split(load_dataset(name, "corpus"))
    queries = _first_split(load_dataset(name, "queries"))
    qrels_ds = load_dataset(name, "default")
    qrels_ds = qrels_ds[qrels_split] if qrels_split in qrels_ds else _first_split(qrels_ds)

    qid_c, did_c = _pick(qrels_ds.column_names, "query-id", "query_id", "qid"), _pick(qrels_ds.column_names, "corpus-id", "corpus_id", "doc_id", "docid")
    sc_c = _pick(qrels_ds.column_names, "score", "relevance", "label")
    qrels: Dict[str, Dict[str, float]] = {}
    for r in qrels_ds:
        if float(r[sc_c]) > 0:
            qrels.setdefault(str(r[qid_c]), {})[str(r[did_c])] = float(r[sc_c])

    q_id_c = _pick(queries.column_names, "_id", "id", "query-id", "query_id")
    q_text_c = _pick(queries.column_names, "text", "query")
    q_map = {str(r[q_id_c]): r[q_text_c] for r in queries if str(r[q_id_c]) in qrels}

    rng = random.Random(seed)
    qids = sorted(q for q in qrels if q in q_map)
    rng.shuffle(qids)
    if n_queries:
        qids = qids[:n_queries]
    needed = {d for q in qids for d in qrels[q]}

    d_id_c = _pick(corpus.column_names, "_id", "id", "doc_id", "corpus-id")
    d_text_c = _pick(corpus.column_names, "text", "body", "passage")
    has_title = with_title and "title" in corpus.column_names
    all_ids = [str(x) for x in corpus[d_id_c]]
    keep = {i for i, d in enumerate(all_ids) if d in needed}
    if n_distractors is None:
        keep = set(range(len(all_ids)))
    else:
        others = [i for i in range(len(all_ids)) if i not in keep]
        keep |= set(rng.sample(others, min(n_distractors, len(others))))
    keep = sorted(keep)
    sub = corpus.select(keep)
    doc_ids = [all_ids[i] for i in keep]
    docs = []
    for r in sub:
        t = r[d_text_c] or ""
        if has_title and r["title"]:
            t = f"{r['title']}. {t}"
        docs.append(normalize_text(t))
    pos = {d: i for i, d in enumerate(doc_ids)}
    return RetrievalData(
        doc_ids=doc_ids, docs=docs, query_ids=qids, queries=[normalize_text(q_map[q]) for q in qids],
        qrels=[{pos[d]: r for d, r in qrels[q].items() if d in pos} for q in qids], name=name,
    )


def toy_dataset() -> RetrievalData:
    """Маленький набор для проверки без интернета: синонимы, многозначность, точные сущности."""
    docs = [
        "Потеряли ключ от входной двери? Мастер вскроет замок и изготовит дубликат ключа за 15 минут.",
        "Ключ шифрования AES длиной 256 бит хранится в аппаратном модуле HSM и регулярно ротируется.",
        "Чтобы записаться к врачу, позвоните по телефону 8 (916) 123-45-67 или напишите на clinic@example.ru.",
        "Договор № 2023/117 от 12.03.2023 заключён между ООО «Ромашка» и Иваном Петровым на сумму 150 000 руб.",
        "Автомобиль не заводится в мороз: проверьте аккумулятор, свечи и стартер.",
        "Машина плохо стартует зимой — чаще всего виновата разряженная батарея.",
        "Рецепт борща: свёкла, капуста, картофель, морковь, лук и говяжий бульон.",
        "Как приготовить украинский суп из свеклы с капустой — пошаговая инструкция.",
        "Москва — столица России, крупнейший по численности населения город страны.",
        "Сбербанк снизил ставки по ипотеке до 9,5 процента годовых с 1 июня 2024 года.",
        "Кредит на жильё стал дешевле: крупнейший банк страны уменьшил процент по займам.",
        "Пароль от Wi-Fi можно найти на наклейке на нижней стороне роутера.",
    ]
    queries = [
        ("где сделать копию ключа от квартиры", {0: 1}),
        ("хранение криптографических ключей", {1: 1}),
        ("+7 916 123 45 67", {2: 1}),
        ("договор 2023/117", {3: 1}),
        ("почему машина не заводится зимой", {4: 1, 5: 1}),
        ("как сварить борщ", {6: 1, 7: 1}),
        ("снижение ставки по ипотеке", {9: 1, 10: 1}),
        ("главный город России", {8: 1}),
    ]
    return RetrievalData(
        doc_ids=[str(i) for i in range(len(docs))], docs=docs,
        query_ids=[f"q{i}" for i in range(len(queries))], queries=[q for q, _ in queries],
        qrels=[r for _, r in queries], name="toy",
    )


# ---------------------------------------------------------------------------
# Поля для гибридного индекса
# ---------------------------------------------------------------------------


def build_fields(texts: Sequence[str], enc: EncodedCorpus, binarizers: Optional[list] = None,
                 use_ner: bool = True) -> Dict[str, List[List[str]]]:
    """Три потока токенов на каждый текст: codes (семантические коды), entities, lemmas."""
    fields = {
        "entities": [extract_entities(t, use_ner=use_ner) for t in texts],
        "lemmas": [lemmatize(t) for t in texts],
    }
    if binarizers:
        fields["codes"] = corpus_code_tokens(enc, binarizers)
    return fields


def coded_text(words: Sequence[str], codes: Sequence[str], entities: Sequence[str], n_hashes: int = 1) -> str:
    """Человекочитаемый вид переписанного текста: слово[код] ... + сущности."""
    parts = []
    for i, w in enumerate(words):
        cs = codes[i * n_hashes: (i + 1) * n_hashes]
        parts.append(f"{w}{''.join(cs)}")
    return " ".join(parts) + ("\n  сущности: " + ", ".join(entities) if entities else "")


def word_code_table(words: Sequence[str], vecs: np.ndarray, binarizer) -> List[Tuple[str, str]]:
    """Пары (слово, код) для демонстрации."""
    return [(w, str(c)) for w, c in zip(words, bits_to_str(binarizer.transform(vecs)))] if len(vecs) else []


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Эксперименты целиком
# ---------------------------------------------------------------------------


def codes_experiment(doc_enc: EncodedCorpus, q_enc: EncodedCorpus, qrels, n_bits: int = 10,
                     n_hashes: int = 1, method: str = "itq", k1: float = 1.2, b: float = 0.75,
                     fit_vecs: Optional[np.ndarray] = None, binarizers: Optional[list] = None,
                     label: str = "", **bin_kw) -> Dict:
    """Одна конфигурация «семантические коды + BM25»: обучение кодов, индекс, метрики, размер, скорость."""
    t0 = time.perf_counter()
    if binarizers is None:
        binarizers = fit_hashes(doc_enc.word_vecs if fit_vecs is None else fit_vecs,
                                n_bits=n_bits, n_hashes=n_hashes, method=method, **bin_kw)
    fit_time = time.perf_counter() - t0
    d_codes = corpus_code_tokens(doc_enc, binarizers)
    q_codes = corpus_code_tokens(q_enc, binarizers)
    bm = BM25(k1, b).fit(d_codes)
    scores, search_time = timed(bm.score_matrix, q_codes)
    return {
        **({"variant": label} if label else {}),
        "method": method, "n_bits": n_bits, "n_hashes": n_hashes, "k1": k1, "b": b,
        **{k: v for k, v in bin_kw.items() if isinstance(v, (int, float, str))},
        **evaluate(scores, qrels),
        "overlap_gap": code_overlap(q_codes, d_codes, qrels)["overlap_gap"],
        "vocab": len(bm.vocab), "index_MB": bm.size_bytes() / 2 ** 20,
        "fit_s": fit_time, "search_ms_per_q": 1000 * search_time / max(len(q_codes), 1),
        "binarizers": binarizers, "scores": scores, "doc_codes": d_codes, "query_codes": q_codes,
    }


def strip_heavy(rows: Iterable[Dict]) -> List[Dict]:
    """Убирает тяжёлые поля (модели, матрицы) перед выводом таблицы."""
    heavy = {"binarizers", "scores", "doc_codes", "query_codes"}
    return [{k: v for k, v in r.items() if k not in heavy} for r in rows]


def show_top(scores: np.ndarray, data: RetrievalData, qi: int, k: int = 5, width: int = 120) -> None:
    """Печатает топ-k документов для запроса qi с пометкой релевантных."""
    print(f"Запрос: {data.queries[qi]}")
    for r, d in enumerate(np.argsort(-scores[qi])[:k], 1):
        mark = "✓" if d in data.qrels[qi] else " "
        print(f"  {r}. {mark} [{scores[qi, d]:.3f}] {data.docs[d][:width]}")
