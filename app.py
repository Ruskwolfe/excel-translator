import io, uuid, os, json, re, unicodedata as ud, difflib, pandas as pd, torch, threading, requests
from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
DICT_URL = "https://kaikki.org/dictionary/Norwegian%20Bokm%C3%A5l/kaikki.org-dictionary-NorwegianBokm%C3%A5l.jsonl"

_dict_idx = None

def fold(s):
    return ud.normalize("NFKC", s).strip()

def fold_key(s):
    return ud.normalize("NFKC", s).strip().casefold()

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

def norm_lang(x):
    x = (x or "").lower().strip()
    if x in ("nb","no"): return "nob_Latn"
    if x == "nn": return "nno_Latn"
    return "eng_Latn"

def get_nllb():
    mid = "facebook/nllb-200-distilled-600M"
    if mid not in _models:
        path = model_path(mid)
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
        _models[mid] = (mdl, tok)
    return _models[mid]

def translate_one(text, src_code, tgt_code, max_new=128, beams=4):
    mdl, tok = get_nllb()
    tok.src_lang = src_code
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(DEVICE) for k,v in enc.items()}
    with torch.no_grad():
        gen = mdl.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code), max_new_tokens=max_new, num_beams=beams, do_sample=False)
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_k(text, src_code, tgt_code, k=5):
    mdl, tok = get_nllb()
    tok.src_lang = src_code
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k2: v.to(DEVICE) for k2,v in enc.items()}
    with torch.no_grad():
        gen = mdl.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code), max_new_tokens=16, num_beams=max(1,k), num_return_sequences=k, do_sample=False, length_penalty=0.8)
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    seen, uniq = set(), []
    for o in outs:
        o2 = o.strip()
        if o2 not in seen:
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

def smart_translate(s, src_code, tgt_code):
    if not one_word(s):
        return translate_one(s, src_code, tgt_code)
    if len(s.strip()) <= 3 and not re.search(r"[A-Za-zÆØÅæøå]", s):
        return s
    d = dict_lookup(s)
    if d:
        return d
    cands = translate_k(s, src_code, tgt_code, k=5)
    if not cands:
        return translate_one(s, src_code, tgt_code)
    best = None
    best_sc = -1.0
    for c in cands:
        try:
            back = translate_one(c, tgt_code, src_code, max_new=16, beams=4)
        except Exception:
            back = ""
        sc = rt_score(s, back) - 0.02 * max(0, len(c.split()) - 1)
        if sc > best_sc:
            best_sc = sc
            best = c
    return best if best is not None else cands[0]

def ensure_nb_dict():
    os.makedirs(DICT_DIR, exist_ok=True)
    if os.path.exists(DICT_IDX):
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
    g = g.strip()
    g = re.sub(r"^\((.*?)\)\s*", "", g)
    g = g.split(";")[0].split(",")[0]
    g = g.replace("to ", "").strip()
    g = re.sub(r"\s+", " ", g)
    return g

def acceptable_gloss(g):
    if not g: return False
    w = g.split()
    if len(w) > 3: return False
    if any(c.isdigit() for c in g): return False
    return True

def extract_best_gloss(obj):
    senses = obj.get("senses") or []
    best = None
    for s in senses:
        gl = s.get("glosses") or []
        if not gl: continue
        g = clean_gloss(gl[0])
        if acceptable_gloss(g):
            if best is None or len(g) < len(best):
                best = g
    return best

def build_nb_index():
    idx = {}
    with open(DICT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            w = obj.get("word")
            if not w: continue
            g = extract_best_gloss(obj)
            if not g: continue
            k = fold_key(w)
            old = idx.get(k)
            if old is None or len(g) < len(old):
                idx[k] = g
    with open(DICT_IDX, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    global _dict_idx
    _dict_idx = idx

def dict_load():
    global _dict_idx
    if _dict_idx is not None:
        return
    if not os.path.exists(DICT_IDX):
        ensure_nb_dict()
    if os.path.exists(DICT_IDX):
        with open(DICT_IDX, "r", encoding="utf-8") as f:
            _dict_idx = json.load(f)
    else:
        _dict_idx = {}

def dict_lookup(s):
    dict_load()
    k = fold_key(s)
    v = _dict_idx.get(k)
    if not v:
        return None
    if s.isupper():
        return v.upper()
    if len(s) > 1 and s[0].isupper() and s[1:].islower():
        return v[:1].upper() + v[1:]
    return v

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
    return {"ready": _ready, "downloading": _downloading, "model": "facebook/nllb-200-distilled-600M", "dict_index_bytes": ds}

@app.get("/dict_test")
def dict_test(term: str):
    try:
        ensure_nb_dict()
        d = dict_lookup(term)
        src = norm_lang("nb")
        tgt = norm_lang("en")
        mt = translate_one(term, src, tgt)
        return {"term": term, "dict": d, "mt": mt}
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
  <div class="notice">Uses Wiktionary via Wiktextract for one-word lookups, then a neural model for everything else</div>
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
  <footer>NLLB-200 distilled 600M + Wiktionary lookup</footer>
</div>
<script src="https://unpkg.com/vue@3"></script>
<script>
const app = Vue.createApp({
  data(){return{file:null,token:null,sheets:[],sheet:null,columns:[],srcCol:null,tgtCol:null,mode:"append_new",srcLang:"nb",tgtLang:"en",status:"",loading:false,downloading:false,timer:null,jobId:null,progressPct:0,done:0,total:0,stage:"",dictFirst:true}}
  ,computed:{readyToTranslate(){return this.token&&this.sheet&&this.srcCol&&this.tgtCol}}
  ,methods:{
    onFile(e){this.file=e.target.files[0];this.token=null;this.sheets=[];this.columns=[];this.srcCol=null;this.tgtCol=null}
    ,async inspect(){if(!this.file)return;this.loading=true;this.status="Inspecting";const fd=new FormData();fd.append("file",this.file);const r=await fetch("/inspect",{method:"POST",body:fd});if(!r.ok){this.status="Failed to read file";this.loading=false;return}const j=await r.json();this.token=j.token;this.sheets=j.sheets;this.sheet=j.sheets[0]||null;this.columns=j.columns||[];this.srcCol=this.columns[0]||null;this.tgtCol=this.srcCol?this.srcCol+"_en":null;this.status="Ready";this.loading=false}
    ,async fetchColumns(){if(!this.token||!this.sheet)return;this.loading=true;this.status="Loading columns";const fd=new FormData();fd.append("token",this.token);fd.append("sheet",this.sheet);const r=await fetch("/columns",{method:"POST",body:fd});if(!r.ok){this.status="Failed to load columns";this.loading=false;return}const j=await r.json();this.columns=j.columns||[];if(!this.srcCol)this.srcCol=this.columns[0]||null;this.status="Ready";this.loading=false}
    ,async start(){this.status="Starting";const fd=new FormData();fd.append("token",this.token);fd.append("sheet",this.sheet);fd.append("src_col",this.srcCol);fd.append("tgt_col",this.tgtCol);fd.append("src_lang",this.srcLang);fd.append("tgt_lang",this.tgtLang);fd.append("mode",this.mode);fd.append("dict_first",this.dictFirst?"true":"false");const r=await fetch("/start",{method:"POST",body:fd});if(!r.ok){this.status="Failed to start";return}const j=await r.json();this.jobId=j.job;this.status="Translating";this.poll()}
    ,async poll(){if(!this.jobId)return;const r=await fetch(`/job?job=${this.jobId}`);if(!r.ok){this.status="Job error";return}const j=await r.json();this.stage=j.stage;this.done=j.done;this.total=j.total;this.downloading=j.stage==="downloading";this.progressPct=j.total?Math.min(100,Math.round(100*j.done/j.total)):(j.stage==="done"?100:0);if(j.stage==="done"){const d=await fetch(`/download?job=${this.jobId}`);const blob=await d.blob();const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="translated.xlsx";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);this.status="Downloaded";this.jobId=null;return}if(j.stage==="error"){this.status=j.error||"Error";this.jobId=null;return}setTimeout(this.poll,500)}
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
        return JSONResponse({"error":"Invalid token"}, status_code=400)
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
            _jobs[job]["stage"]="error"; _jobs[job]["error"]="Missing source column"; return
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
        for i,(s,t) in enumerate(zip(src_vals, tgt_vals)):
            s2 = s.strip()
            if s2 == "":
                continue
            if s2.startswith("="):
                continue
            if mode == "skip_filled" and t.strip() != "":
                continue
            if mode == "append_new" and t.strip() != "":
                continue
            if s2 not in cache:
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
                    if dict_first and one_word(s):
                        d = dict_lookup(s)
                    else:
                        d = None
                    out = d if d else smart_translate(s, src_code, tgt_code)
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
        return JSONResponse({"error":"Invalid token"}, status_code=400)
    j = str(uuid.uuid4())
    _jobs[j] = {"stage":"queued","done":0,"total":0}
    use_dict = str(dict_first).lower() == "true"
    t = threading.Thread(target=run_job, args=(j, token, sheet, src_col, tgt_col, src_lang, tgt_lang, mode, use_dict), daemon=True)
    t.start()
    return {"job": j}

@app.get("/job")
async def job(job: str = Query(...)):
    if job not in _jobs:
        return JSONResponse({"error":"Unknown job"}, status_code=404)
    d = dict(_jobs[job])
    d.pop("result", None)
    return d

@app.get("/download")
async def download(job: str = Query(...)):
    if job not in _jobs or _jobs[job].get("stage") != "done":
        return JSONResponse({"error":"Not ready"}, status_code=400)
    data = _jobs[job]["result"]
    return StreamingResponse(io.BytesIO(data), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition":"attachment; filename=translated.xlsx"})
