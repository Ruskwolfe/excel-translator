import io, uuid, os, json, re, unicodedata as ud, difflib, pandas as pd, torch, threading, requests
from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

app = FastAPI()
_files = {}
_models = {}
_jobs = {}
_ready = False
_downloading = False
_cache_path = os.environ.get("MT_CACHE_PATH", ".translate_cache.json")
MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DICT_DIR = os.environ.get("DICT_DIR", "dict")
DICT_JSONL = os.path.join(DICT_DIR, "nb.jsonl")
DICT_IDX = os.path.join(DICT_DIR, "nb_idx.json")
DICT_ASC_IDX = os.path.join(DICT_DIR, "nb_idx_ascii.json")
DICT_URL = "https://kaikki.org/dictionary/Norwegian%20Bokm%C3%A5l/kaikki.org-dictionary-NorwegianBokm%C3%A5l.jsonl"
ART_RE = re.compile(r'^(?:an?|the)\s+', re.I)
PARENS_RE = re.compile(r'\([^)]*\)')
HEADS = {"skive","hjul","lampe","mal","senter","styring","sliper","skjerm","hylle","mutter","maskin","hus","boks","kabel","plugg","bolt","skrue","ror","ventil","kontakt","sensor","pumpe","motor","plate","holder","verktoy","ledning","panel"}
BAD_FINAL = {"smal","stor","liten","god","ny","fri"}

_dict_idx = None
_dict_idx_ascii = None

def fold(s):
    return ud.normalize("NFKC", s).strip()

def fold_key(s):
    return ud.normalize("NFKC", s).strip().casefold()

def fold_ascii(s):
    d = ud.normalize("NFKD", s)
    return "".join(c for c in d if not ud.combining(c)).casefold().strip()

def load_cache():
    if os.path.exists(_cache_path):
        with open(_cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(c):
    with open(_cache_path, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)

def model_path(mid):
    local = os.path.join(MODEL_ROOT, *mid.split("/"))
    return local if os.path.isdir(local) else mid

_ban_cache = {}

class BanAtStart(LogitsProcessor):
    def __init__(self, ids):
        self.ids = set(i for i in ids if i is not None)
    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == 1 and self.ids:
            scores[:, list(self.ids)] = -1e9
        return scores

class BanAlways(LogitsProcessor):
    def __init__(self, ids):
        self.ids = set(i for i in ids if i is not None)
    def __call__(self, input_ids, scores):
        if self.ids:
            scores[:, list(self.ids)] = -1e9
        return scores

def build_bans(tok):
    start = []
    for s in [" a", " an", " the"]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            start.append(ids[0])
    anyt = []
    for s in ["(", " (", ")", " )"]:
        ids = tok.encode(s, add_special_tokens=False)
        anyt.extend(ids)
    return list(dict.fromkeys(start)), list(dict.fromkeys(anyt))

def get_logits_processors(tok):
    k = id(tok)
    if k not in _ban_cache:
        s, a = build_bans(tok)
        _ban_cache[k] = LogitsProcessorList([BanAtStart(s), BanAlways(a)])
    return _ban_cache[k]

def norm_lang(x):
    x = (x or "").lower().strip()
    if x in ("nb", "no"):
        return "nob_Latn"
    if x == "nn":
        return "nno_Latn"
    return "eng_Latn"

def get_nllb():
    mid = os.environ.get("MT_MODEL_ID", "facebook/nllb-200-distilled-600M")
    if mid not in _models:
        path = model_path(mid)
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
        _models[mid] = (mdl, tok)
    return _models[mid]

def translate_one(text, src_code, tgt_code, max_new=64, beams=4):
    mdl, tok = get_nllb()
    tok.src_lang = src_code
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        gen = mdl.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
            max_new_tokens=max_new,
            num_beams=beams,
            do_sample=False,
            logits_processor=get_logits_processors(tok),
        )
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_k(text, src_code, tgt_code, k=12):
    mdl, tok = get_nllb()
    tok.src_lang = src_code
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k2: v.to(DEVICE) for k2, v in enc.items()}
    with torch.no_grad():
        gen = mdl.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
            max_new_tokens=16,
            num_beams=max(8, k),
            num_return_sequences=min(k, max(8, k)),
            do_sample=False,
            length_penalty=1.0,
            num_beam_groups=4,
            diversity_penalty=0.2,
            logits_processor=get_logits_processors(tok),
        )
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    seen, uniq = set(), []
    for o in outs:
        o2 = o.strip()
        if o2 and o2 not in seen:
            seen.add(o2)
            uniq.append(o2)
    return uniq

def one_word(s):
    core = re.sub(r"^[\W_]+|[\W_]+$", "", s.strip())
    return core != "" and " " not in core

def rt_score(src, back):
    a = fold(src).casefold()
    b = fold(back).casefold()
    s1 = difflib.SequenceMatcher(None, a, b).ratio()
    a2 = ud.normalize("NFKD", src)
    a2 = "".join(c for c in a2 if not ud.combining(c)).casefold()
    b2 = ud.normalize("NFKD", back)
    b2 = "".join(c for c in b2 if not ud.combining(c)).casefold()
    s2 = difflib.SequenceMatcher(None, a2, b2).ratio()
    return max(s1, s2)

def ensure_nb_dict():
    os.makedirs(DICT_DIR, exist_ok=True)
    if os.path.exists(DICT_IDX) and os.path.exists(DICT_ASC_IDX):
        return
    if not os.path.exists(DICT_JSONL):
        with requests.get(DICT_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(DICT_JSONL, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
    build_nb_index()

def clean_gloss(g):
    g = PARENS_RE.sub('', g or '')
    g = g.replace(';', ',').split(',')[0]
    g = ART_RE.sub('', g)
    g = re.sub(r'\s+', ' ', g).strip()
    return g

def acceptable_gloss(g):
    if not g:
        return False
    wl = g.split()
    if len(wl) > 3:
        return False
    if any(c.isdigit() for c in g):
        return False
    bad = {"party", "festival", "feast", "banquet"}
    if g.lower() in bad:
        return False
    if "chemical element" in g.lower():
        return False
    return True

def extract_best_translation(obj):
    pos_order = ["noun", "adjective", "verb", "adverb", "pronoun", "numeral", "name"]
    senses = obj.get("senses") or []
    best = None
    best_rank = (10, 999)
    for s in senses:
        p = (s.get("pos") or "").lower()
        if p == "proper-noun":
            continue
        if s.get("form_of"):
            continue
        trs = s.get("translations") or []
        for t in trs:
            if (t.get("lang") or "").lower() != "english":
                continue
            tags = [x.lower() for x in (t.get("tags") or [])]
            if any(x in {"name", "proper"} for x in tags):
                continue
            w = clean_gloss(t.get("word") or "")
            if not acceptable_gloss(w):
                continue
            rank = pos_order.index(p) if p in pos_order else len(pos_order) + 1
            key = (rank, len(w))
            if key < best_rank:
                best = w
                best_rank = key
        if best:
            continue
        gl = s.get("glosses") or []
        if not gl:
            continue
        g = clean_gloss(gl[0])
        if not acceptable_gloss(g):
            continue
        rank = pos_order.index(p) if p in pos_order else len(pos_order) + 1
        key = (rank, len(g))
        if key < best_rank:
            best = g
            best_rank = key
    return best

def build_nb_index():
    lemma = {}
    aliases = {}
    with open(DICT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            w = obj.get("word")
            if not w:
                continue
            k = fold_key(w)
            g = extract_best_translation(obj)
            if g:
                prev = lemma.get(k)
                if prev is None or len(g) < len(prev):
                    lemma[k] = g
            senses = obj.get("senses") or []
            for s in senses:
                fos = s.get("form_of") or []
                for fo in fos:
                    b = fo.get("word")
                    if b:
                        ak = k
                        bk = fold_key(b)
                        aliases.setdefault(ak, set()).add(bk)
    idx = {}
    for k, g in lemma.items():
        idx[k] = g
    for ak, bases in aliases.items():
        if ak in idx:
            continue
        for bk in bases:
            if bk in lemma:
                idx[ak] = lemma[bk]
                break
    asc = {}
    for k, g in idx.items():
        a = fold_ascii(k)
        if a and (a not in asc or len(g) < len(asc[a])):
            asc[a] = g
    with open(DICT_IDX, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    with open(DICT_ASC_IDX, "w", encoding="utf-8") as f:
        json.dump(asc, f, ensure_ascii=False)
    global _dict_idx, _dict_idx_ascii
    _dict_idx = idx
    _dict_idx_ascii = asc

def dict_load():
    global _dict_idx, _dict_idx_ascii
    if _dict_idx is not None and _dict_idx_ascii is not None:
        return
    if not os.path.exists(DICT_IDX) or not os.path.exists(DICT_ASC_IDX):
        ensure_nb_dict()
    if os.path.exists(DICT_IDX):
        with open(DICT_IDX, "r", encoding="utf-8") as f:
            _dict_idx = json.load(f)
    else:
        _dict_idx = {}
    if os.path.exists(DICT_ASC_IDX):
        with open(DICT_ASC_IDX, "r", encoding="utf-8") as f:
            _dict_idx_ascii = json.load(f)
    else:
        _dict_idx_ascii = {}

def dict_lookup_base_ascii_key(k):
    v = _dict_idx.get(k)
    if not v:
        a = fold_ascii(k)
        v = _dict_idx_ascii.get(a)
    return v

def dict_lookup(s):
    dict_load()
    k = fold_key(s)
    v = dict_lookup_base_ascii_key(k)
    if not v:
        return None
    if s.isupper():
        return v.upper()
    if len(s) > 1 and s[0].isupper() and s[1:].islower():
        return v[:1].upper() + v[1:]
    return v

def is_safe_english_noun(v):
    return v.isalpha() and v == v.lower() and len(v) <= 24

def dict_fuzzy(s, cutoff=None):
    dict_load()
    key = fold_ascii(s)
    c = 0.93 if cutoff is None else cutoff
    if len(key) >= 8:
        c = min(c, 0.89)
    matches = difflib.get_close_matches(key, list(_dict_idx_ascii.keys()), n=1, cutoff=c)
    if not matches:
        return None
    cand = matches[0]
    if len(key) <= 6 and (key[0] != cand[0] or key[-1] != cand[-1]):
        return None
    v = _dict_idx_ascii.get(cand)
    if not v or not is_safe_english_noun(v):
        return None
    if s.isupper():
        return v.upper()
    if len(s) > 1 and s[0].isupper() and s[1:].islower():
        return v[:1].upper() + v[1:]
    return v

def ascii_close(seg, cutoff):
    m = difflib.get_close_matches(seg, list(_dict_idx_ascii.keys()), n=1, cutoff=cutoff)
    if not m:
        return None
    cand = m[0]
    if len(seg) >= 5 and (seg[0] != cand[0] or seg[-1] != cand[-1]):
        return None
    if abs(len(cand) - len(seg)) > 2:
        return None
    return cand

def seg_score(seg):
    if seg in (_dict_idx or {}):
        return 0.0, seg, True
    if seg in (_dict_idx_ascii or {}):
        return 0.0, seg, True
    if seg.endswith('s') and seg[:-1] in (_dict_idx or {}):
        return 0.1, seg[:-1], True
    if seg.endswith('s') and seg[:-1] in (_dict_idx_ascii or {}):
        return 0.1, seg[:-1], True
    cm = ascii_close(seg, 0.9 if len(seg) >= 6 else 0.94)
    if cm:
        return 0.7, cm, False
    if len(seg) <= 4:
        return 1.5, seg, False
    return 3.0, seg, False

def strip_leading_article(w):
    return ART_RE.sub('', w).strip()

def contains_untranslated_upper_token(src, cand):
    src_toks = re.findall(r'\b[A-ZÆØÅ]{2,4}\b', src)
    if not src_toks:
        return False
    for t in src_toks:
        if re.search(r'\b' + re.escape(t) + r'\b', cand):
            return True
    return False

def cand_penalty(c, src):
    p = 0.0
    if re.match(r'(?i)^\s*(?:a|an|the)\s+', c):
        p -= 0.4
    if '(' in c or ')' in c:
        p -= 0.4
    if len(c.split()) > 3:
        p -= 0.2
    if re.search(r'\d', c):
        p -= 0.1
    if re.match(r'^[A-Z][a-z]+$', c):
        p -= 0.2
    if re.search(r'(?i)(\b\d+\s*mm\b|\bM\d)', src) and re.search(r'(?i)\bmother\b', c):
        p -= 0.6
    if re.fullmatch(r'[A-Za-z]+', c) and difflib.SequenceMatcher(None, fold_ascii(src), c.lower()).ratio() >= 0.8:
        p -= 0.5
    if re.search(r'(?i)\bto\b', c):
        p -= 0.35
    if contains_untranslated_upper_token(src, c):
        p -= 0.6
    if fold_ascii(c) == fold_ascii(src):
        p -= 0.8
    return p

def nounify_head(w):
    m = re.match(r'(?i)^(?:to\s+)?([a-z]+)$', w)
    if not m:
        return {w}
    v = m.group(1)
    return {w, v + "er"}

def compound_candidate(segs, src_code, tgt_code):
    pieces = []
    for i, seg in enumerate(segs):
        base_en = (_dict_idx or {}).get(seg) or (_dict_idx_ascii or {}).get(seg)
        if not base_en:
            dtry = dict_lookup(seg)
            base_en = dtry if dtry else strip_leading_article(translate_one(seg, src_code, tgt_code, max_new=8, beams=4))
        if i == len(segs) - 1:
            heads = nounify_head(base_en)
            return [" ".join(pieces + [h]) for h in heads]
        pieces.append(base_en)
    return [" ".join(pieces)]

def split_compound_dp(y, max_seg=24):
    n = len(y)
    dp = [None] * (n + 1)
    dp[0] = (0.0, [])
    exact_mask = [None] * (n + 1)
    exact_mask[0] = []
    for i in range(n):
        if dp[i] is None:
            continue
        limit = min(n, i + max_seg)
        for j in range(i + 2, limit + 1):
            seg = y[i:j]
            if not seg.isalpha():
                continue
            cost, base, exact = seg_score(seg)
            prev_cost, prev_segs = dp[i]
            adj = 0.0
            if j == n:
                if base in HEADS:
                    adj -= 0.35
                if base in BAD_FINAL:
                    adj += 0.6
            cand_cost = prev_cost + cost + 0.05 + adj
            cand_list = prev_segs + [base]
            if dp[j] is None or cand_cost < dp[j][0]:
                dp[j] = (cand_cost, cand_list)
                exact_mask[j] = (exact_mask[i] or []) + [exact]
    if dp[n] is None:
        return None, False
    cost, segs = dp[n]
    exact = all(exact_mask[n]) if exact_mask[n] else False
    if len(segs) < 2:
        return None, False
    return segs, exact

def split_compound_suffix(y):
    for h in sorted(HEADS, key=len, reverse=True):
        if y.endswith(h) and len(y) > len(h) + 1:
            return [y[: len(y) - len(h)], h], False
    return None, False

def split_compound(w):
    dict_load()
    x = fold_key(w)
    y = fold_ascii(w)
    n1, e1 = split_compound_dp(x)
    n2, e2 = split_compound_dp(y)
    if n1 and n2:
        if e1 and not e2:
            return n1, e1
        if e2 and not e1:
            return n2, e2
        if len("".join(n1)) >= len("".join(n2)):
            return n1, e1
        return n2, e2
    if n1:
        return n1, e1
    if n2:
        return n2, e2
    n3, e3 = split_compound_suffix(y)
    if n3:
        return n3, e3
    return None, False

def is_caps_phrase(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

def is_code_token(tok):
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9\-/\.]*', tok):
        return False
    if tok.isalpha():
        return False
    return fold_ascii(tok) == tok.lower()

def en_tweaks(x):
    x = re.sub(r'\b([Ss]ilver)\s+plated\b', r'\1-plated', x)
    x = re.sub(r'\bheat cable\b', 'heating cable', x)
    x = re.sub(r'\bventilation house\b', 'ventilation housing', x)
    x = re.sub(r'\bcapsling\b', 'enclosure', x)
    x = re.sub(r'\bcapsing\b', 'enclosure', x)
    return x

def post_norm(out, src):
    x = out
    x = re.sub(r'\([^)]*\)', '', x)
    x = re.sub(r'(?i)\b(\d{1,6})\s*grader\b', r'\1°', x)
    x = re.sub(r'(?i)\b(\d{1,6})\s*gr\b', r'\1°', x)
    x = re.sub(r'(?i)\b(\d{1,6})\s*mm\b', r'\1mm', x)
    x = re.sub(r'(?i)\bnr\.?\b', 'No.', x)
    x = re.sub(r'No\.\.', 'No.', x)
    x = re.sub(r'(\d),(\d)', r'\1.\2', x)
    x = re.sub(r'\s+', ' ', x).strip()
    x = re.sub(r'(?i)\b(a|an)\s+(a|an)\s+', r'\1 ', x)
    x = re.sub(r'(?i)^(?:a|an|the)\s+(?=[a-z])', '', x)
    x = en_tweaks(x)
    if is_caps_phrase(src):
        x = x.upper()
    return x

def translate_phrase(s, src_code, tgt_code, dict_first=True):
    parts = re.split(r'(\W+)', s)
    out = []
    for tok in parts:
        if tok == "":
            continue
        if re.fullmatch(r'[^\W\d_]+', tok, re.UNICODE):
            if is_code_token(tok):
                out.append(tok)
                continue
            t = None
            if dict_first:
                t = dict_lookup(tok)
                if not t:
                    comp, exact = split_compound(tok)
                    if comp:
                        pieces = []
                        ok = True
                        for seg in comp:
                            base_en = (_dict_idx or {}).get(seg) or (_dict_idx_ascii or {}).get(seg)
                            if not base_en:
                                ok = False
                                break
                            pieces.append(base_en)
                        if ok:
                            t = " ".join(pieces)
                if not t:
                    tf = dict_fuzzy(tok)
                    if tf:
                        t = tf
            if not t:
                if one_word(tok):
                    t = translate_one(tok, src_code, tgt_code)
                else:
                    t = tok
            if one_word(tok):
                t = strip_leading_article(t)
            out.append(t)
        else:
            out.append(tok)
    wordwise = post_norm("".join(out), s)
    mt_all = post_norm(translate_one(s, src_code, tgt_code), s)
    try:
        back_word = translate_one(wordwise, tgt_code, src_code, max_new=16, beams=4)
    except Exception:
        back_word = ""
    try:
        back_mt = translate_one(mt_all, tgt_code, src_code, max_new=16, beams=4)
    except Exception:
        back_mt = ""
    sc_word = rt_score(s, back_word) + cand_penalty(wordwise, s)
    sc_mt = rt_score(s, back_mt) + cand_penalty(mt_all, s)
    return wordwise if sc_word >= sc_mt else mt_all

def smart_translate(s, src_code, tgt_code, dict_first=True):
    if " " in s.strip():
        return translate_phrase(s, src_code, tgt_code, dict_first)
    if not one_word(s):
        return post_norm(translate_one(s, src_code, tgt_code), s)
    if len(s.strip()) <= 3 and not re.search(r"[A-Za-zÆØÅæøå]", s):
        return s
    dict_cand = None
    comp = None
    exact = False
    if dict_first:
        dict_cand = dict_lookup(s)
        if not dict_cand:
            comp, exact = split_compound(s)
            if comp:
                try:
                    dict_cand = " ".join(((_dict_idx or {}).get(seg) or (_dict_idx_ascii or {}).get(seg) or "") for seg in comp)
                    if "" in dict_cand:
                        dict_cand = None
                except Exception:
                    dict_cand = None
        if not dict_cand:
            dict_cand = dict_fuzzy(s)
    cands = []
    if dict_cand:
        cands.append(strip_leading_article(dict_cand))
    if comp:
        try:
            cands.extend(compound_candidate(comp, src_code, tgt_code))
        except Exception:
            pass
    mt = translate_k(s, src_code, tgt_code, k=12) or [translate_one(s, src_code, tgt_code)]
    for c in mt:
        if one_word(s):
            c = strip_leading_article(c)
        cands.append(c)
    best = None
    best_sc = -1e9
    for c in dict.fromkeys([post_norm(x, s) for x in cands if x]):
        try:
            back = translate_one(c, tgt_code, src_code, max_new=8, beams=4)
        except Exception:
            back = ""
        sc = rt_score(s, back) + cand_penalty(c, s)
        if dict_cand and c.lower() == strip_leading_article(dict_cand).lower():
            sc += 0.8
        if sc > best_sc:
            best_sc = sc
            best = c
    return post_norm(best if best is not None else cands[0], s)

def preload():
    global _ready, _downloading
    if _ready:
        return
    _downloading = True
    try:
        translate_one("hei", "nob_Latn", "eng_Latn")
        _ready = True
    finally:
        _downloading = False

@app.on_event("startup")
def on_start():
    threading.Thread(target=preload, daemon=True).start()

@app.get("/status")
def status():
    ds = os.path.getsize(DICT_IDX) if os.path.exists(DICT_IDX) else 0
    mid = os.environ.get("MT_MODEL_ID", "facebook/nllb-200-distilled-600M")
    return {"ready": _ready, "downloading": _downloading, "model": mid, "dict_index_bytes": ds}

@app.post("/clear_cache")
def clear_cache():
    try:
        if os.path.exists(_cache_path):
            os.remove(_cache_path)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/dict_test")
def dict_test(term: str):
    try:
        ensure_nb_dict()
        d = dict_lookup(term)
        src = norm_lang("nb")
        tgt = norm_lang("en")
        mt = translate_one(term, src, tgt)
        comp, exact = split_compound(term)
        fuzz = dict_fuzzy(term) if not d else None
        return {"term": term, "dict": d, "compound": comp, "compound_exact": exact, "fuzzy": fuzz, "mt": mt}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Excel Column Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0b0d10;color:#e8eef5;margin:0}
.container{max-width:900px;margin:40px auto;padding:24px;background:#12161b;border-radius:18px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
h1{font-size:24px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.row{display:flex;gap:12px;align-items:center;margin:12px 0}
.card{background:#0f1317;border:1px solid #1d232b;border-radius:14px;padding:16px}
label{font-size:13px;color:#a7b0bc}
input[type=file],select,button,input[type=text]{width:100%}
select,input[type=file],input[type=text]{background:#0b0f13;border:1px solid #27303b;color:#e8eef5;border-radius:12px;padding:10px;font-size:14px}
button{background:#2f6feb;border:0;color:#fff;border-radius:12px;padding:12px 14px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
small{color:#97a3b2}
hr{border:0;border-top:1px solid #1d232b;margin:16px 0}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#1a2130;border:1px solid #27303b;font-size:12px}
.notice{background:#151b25;border:1px solid #27303b;padding:10px 12px;border-radius:10px;margin:12px 0;font-size:14px}
.progress{height:10px;background:#0b0f13;border:1px solid #27303b;border-radius:999px;overflow:hidden}
.bar{height:100%;width:0%}
footer{opacity:.7;font-size:12px;margin-top:12px}
</style>
</head>
<body>
<div id="app" class="container">
  <h1>Excel Column Translator</h1>
  <div class="notice">Wiktextract EN translations, DP compound split with fuzzy bounds, codes and units preserved</div>
  <div class="card">
    <div class="row">
      <input type="file" @change="onFile" accept=".xlsx,.xls" />
      <button :disabled="!file || loading" @click="inspect">Load</button>
    </div>
    <div v-if="token">
      <div class="grid">
        <div>
          <label>Sheet</label>
          <select v-model="sheet" @change="fetchColumns">
            <option v-for="s in sheets" :key="s" :value="s">{{s}}</option>
          </select>
        </div>
        <div>
          <label>Source column</label>
          <select v-model="srcCol">
            <option v-for="c in columns" :key="c" :value="c">{{c}}</option>
          </select>
        </div>
      </div>
      <div class="grid" style="margin-top:12px">
        <div>
          <label>Target column</label>
          <input list="cols" v-model="tgtCol" placeholder="Type or pick a column">
          <datalist id="cols"><option v-for="c in columns" :key="c" :value="c"></option></datalist>
        </div>
        <div>
          <label>Mode</label>
          <select v-model="mode">
            <option value="append_new">Create if missing</option>
            <option value="overwrite">Overwrite target values</option>
            <option value="skip_filled">Skip rows with target text</option>
          </select>
        </div>
      </div>
      <div class="row" style="margin-top:12px">
        <label style="display:flex;align-items:center;gap:8px"><input type="checkbox" v-model="dictFirst" style="width:auto">Use dictionary first for single words</label>
      </div>
      <hr>
      <div class="row" v-if="jobId">
        <div class="progress" style="flex:1"><div class="bar" :style="{width: progressPct+'%', background: downloading? '#444' : '#2f6feb'}"></div></div>
        <div class="badge">{{stage}} {{done}}/{{total}}</div>
      </div>
      <div class="row">
        <button :disabled="!readyToTranslate || loading || jobId" @click="start">Translate</button>
        <div class="badge" v-if="status">{{status}}</div>
      </div>
      <small>Source: Wiktionary via Wiktextract on kaikki.org</small>
    </div>
  </div>
  <footer>NLLB-200 600M + Wiktextract + DP splitter</footer>
</div>
<script src="https://unpkg.com/vue@3"></script>
<script>
const app = Vue.createApp({
  data(){return{file:null,token:null,sheets:[],sheet:null,columns:[],srcCol:null,tgtCol:null,mode:"append_new",srcLang:"nb",tgtLang:"en",status:"",loading:false,downloading:false,timer:null,jobId:null,progressPct:0,done:0,total:0,stage:"",dictFirst:true}},
  computed:{readyToTranslate(){return this.token&&this.sheet&&this.srcCol&&this.tgtCol}},
  methods:{
    onFile(e){this.file=e.target.files[0];this.token=null;this.sheets=[];this.columns=[];this.srcCol=null;this.tgtCol=null},
    async inspect(){if(!this.file)return;this.loading=true;this.status="Inspecting";const fd=new FormData();fd.append("file",this.file);const r=await fetch("/inspect",{method:"POST",body:fd});if(!r.ok){this.status="Failed to read file";this.loading=false;return}const j=await r.json();this.token=j.token;this.sheets=j.sheets;this.sheet=j.sheets[0]||null;this.columns=j.columns||[];this.srcCol=this.columns[0]||null;this.tgtCol=this.srcCol?this.srcCol+"_en":null;this.status="Ready";this.loading=false},
    async fetchColumns(){if(!this.token||!this.sheet)return;this.loading=true;this.status="Loading columns";const fd=new FormData();fd.append("token",this.token);fd.append("sheet",this.sheet);const r=await fetch("/columns",{method:"POST",body:fd});if(!r.ok){this.status="Failed to load columns";this.loading=false;return}const j=await r.json();this.columns=j.columns||[];if(!this.srcCol)this.srcCol=this.columns[0]||null;this.status="Ready";this.loading=false},
    async start(){this.status="Starting";const fd=new FormData();fd.append("token",this.token);fd.append("sheet",this.sheet);fd.append("src_col",this.srcCol);fd.append("tgt_col",this.tgtCol);fd.append("src_lang",this.srcLang);fd.append("tgt_lang",this.tgtLang);fd.append("mode",this.mode);fd.append("dict_first",this.dictFirst?"true":"false");const r=await fetch("/start",{method:"POST",body:fd});if(!r.ok){this.status="Failed to start";return}const j=await r.json();this.jobId=j.job;this.status="Translating";this.poll()},
    async poll(){if(!this.jobId)return;const r=await fetch(`/job?job=${this.jobId}`);if(!r.ok){this.status="Job error";return}const j=await r.json();this.stage=j.stage;this.done=j.done;this.total=j.total;this.downloading=j.stage==="downloading";this.progressPct=j.total?Math.min(100,Math.round(100*j.done/j.total)):(j.stage==="done"?100:0);if(j.stage==="done"){const d=await fetch(`/download?job=${this.jobId}`);const blob=await d.blob();const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="translated.xlsx";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);this.status="Downloaded";this.jobId=null;return}if(j.stage==="error"){this.status=j.error||"Error";this.jobId=null;return}setTimeout(this.poll,500)}
  }
})
app.mount("#app")
</script>
</body>
</html>
"""

@app.post("/inspect")
async def inspect(file: UploadFile):
    data = await file.read()
    tok = str(uuid.uuid4())
    _files[tok] = data
    try:
        xf = pd.ExcelFile(io.BytesIO(data))
        sheets = xf.sheet_names
        first = sheets[0] if sheets else None
        cols = []
        if first:
            df = xf.parse(first, nrows=1)
            cols = list(df.columns.astype(str))
        return JSONResponse({"token": tok, "sheets": sheets, "columns": cols})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/columns")
async def columns(token: str = Form(...), sheet: str = Form(...)):
    if token not in _files:
        return JSONResponse({"error": "Invalid token"}, status_code=400)
    try:
        xf = pd.ExcelFile(io.BytesIO(_files[token]))
        df = xf.parse(sheet, nrows=1)
        cols = list(df.columns.astype(str))
        return JSONResponse({"columns": cols})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

def run_job(job, token, sheet, src_col, tgt_col, src_lang, tgt_lang, mode, dict_first):
    try:
        _jobs[job]["stage"] = "loading"
        xf = pd.ExcelFile(io.BytesIO(_files[token]))
        df = xf.parse(sheet)
        if src_col not in df.columns:
            _jobs[job]["stage"] = "error"
            _jobs[job]["error"] = "Missing source column"
            return
        if tgt_col not in df.columns:
            df[tgt_col] = ""
        else:
            df[tgt_col] = df[tgt_col].fillna("")
        if dict_first:
            _jobs[job]["stage"] = "downloading"
            try:
                ensure_nb_dict()
            except Exception as e:
                _jobs[job]["dict_error"] = str(e)
        cache = load_cache()
        src_vals = df[src_col].fillna("").astype(str)
        tgt_vals = df[tgt_col].fillna("").astype(str)
        todo_idx = []
        todo_texts = []
        for i, (s, t) in enumerate(zip(src_vals, tgt_vals)):
            s2 = s.strip()
            if s2 == "":
                continue
            if s2.startswith("="):
                continue
            if mode == "skip_filled" and t.strip() != "":
                continue
            if mode == "append_new" and t.strip() != "":
                continue
            if dict_first and one_word(s2):
                todo_texts.append(s2)
            elif s2 not in cache:
                todo_texts.append(s2)
            todo_idx.append(i)
        uniq = list(dict.fromkeys(todo_texts))
        _jobs[job]["total"] = max(len(uniq), 1)
        if uniq:
            _jobs[job]["stage"] = "translating"
            src_code = norm_lang(src_lang)
            tgt_code = norm_lang(tgt_lang)
            for i, s in enumerate(uniq, 1):
                try:
                    d = dict_lookup(s) if dict_first and one_word(s) else None
                    if not d and dict_first and one_word(s):
                        comp, exact = split_compound(s)
                        if comp and exact:
                            d = " ".join(((_dict_idx or {}).get(seg) or (_dict_idx_ascii or {}).get(seg) or seg) for seg in comp)
                    out = d if d else smart_translate(s, src_code, tgt_code, dict_first)
                    cache[s] = out
                except Exception:
                    cache[s] = ""
                _jobs[job]["done"] = i
            save_cache(cache)
        _jobs[job]["stage"] = "writing"
        for i in todo_idx:
            s2 = src_vals.iat[i].strip()
            df.at[i, tgt_col] = cache.get(s2, "")
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            for sh in xf.sheet_names:
                if sh == sheet:
                    df.to_excel(writer, sheet_name=sh, index=False)
                else:
                    xf.parse(sh).to_excel(writer, sheet_name=sh, index=False)
        bio.seek(0)
        _jobs[job]["result"] = bio.read()
        _jobs[job]["stage"] = "done"
    except Exception as e:
        _jobs[job]["stage"] = "error"
        _jobs[job]["error"] = str(e)

@app.post("/start")
async def start(token: str = Form(...), sheet: str = Form(...), src_col: str = Form(...), tgt_col: str = Form(...), src_lang: str = Form(...), tgt_lang: str = Form(...), mode: str = Form("append_new"), dict_first: str = Form("true")):
    if token not in _files:
        return JSONResponse({"error": "Invalid token"}, status_code=400)
    j = str(uuid.uuid4())
    _jobs[j] = {"stage": "queued", "done": 0, "total": 0}
    use_dict = str(dict_first).lower() == "true"
    t = threading.Thread(target=run_job, args=(j, token, sheet, src_col, tgt_col, src_lang, tgt_lang, mode, use_dict), daemon=True)
    t.start()
    return {"job": j}

@app.get("/job")
async def job(job: str = Query(...)):
    if job not in _jobs:
        return JSONResponse({"error": "Unknown job"}, status_code=404)
    d = dict(_jobs[job])
    d.pop("result", None)
    return d

@app.get("/download")
async def download(job: str = Query(...)):
    if job not in _jobs or _jobs[job].get("stage") != "done":
        return JSONResponse({"error": "Not ready"}, status_code=400)
    data = _jobs[job]["result"]
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=translated.xlsx"},
    )
