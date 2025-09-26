import io, uuid, os, json, re, pandas as pd, torch, threading
from fastapi import FastAPI, UploadFile, Form, Query, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from contextlib import asynccontextmanager

_files = {}
_jobs = {}
_models = {}
_glossaries = {}
_ready = False

CACHE_PATH = os.environ.get("MT_CACHE_PATH", ".translate_cache.json")
MODEL_ROOT = os.environ.get("MODEL_ROOT", "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAST_BATCH = int(os.environ.get("MT_BATCH", "128"))
MASK_CODES = os.environ.get("MASK_CODES", "1") == "1"
CACHE_VER = "v3"

PLACE_OPEN = "｟"
PLACE_CLOSE = "｠"

_special_tok_ready = {}
CODE2_RE = re.compile(r"(?<!\w)(?:[A-Za-zÆØÅæøå]+[A-Za-z0-9ÆØÅæøå\-/\.]*\d[A-Za-z0-9ÆØÅæøå\-/\.]*|\d[A-Za-zÆØÅæøå][A-Za-z0-9ÆØÅæøå\-/\.]*)(?!\w)")
RESTORE_RE = re.compile(r"｟\s*c\s*(\d+)\s*｠", re.I)

def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(c):
    try:
        json.dump(c, open(CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass

def model_path(mid):
    p = os.path.join(MODEL_ROOT, *mid.split("/"))
    return p if os.path.isdir(p) else mid

def get_model_ids():
    v = os.environ.get("MT_MODEL_IDS", "").strip()
    if v:
        return [x.strip() for x in v.split(",") if x.strip()]
    return [
        "facebook/nllb-200-1.3B" if DEVICE == "cuda" else "facebook/nllb-200-distilled-600M",
        "facebook/m2m100_1.2B",
        "Helsinki-NLP/opus-mt-tc-big-gmq-en"
    ]

def load_model(mid):
    path = model_path(mid)
    tok = AutoTokenizer.from_pretrained(path, token=False)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(path, token=False)
    if DEVICE == "cuda":
        try:
            mdl = mdl.half()
        except Exception:
            pass
    mdl = mdl.to(DEVICE)
    return mdl, tok

def get_model(mid=None):
    mids = get_model_ids()
    pick = mid or mids[0]
    if pick not in _models:
        _models[pick] = load_model(pick)
    return pick, *_models[pick]

def pick_model_for_target(model_id, tgt_ui):
    if model_id:
        return get_model(model_id)
    mids = get_model_ids()
    for m in mids:
        if "helsinki-nlp/opus-mt" not in m.lower():
            return get_model(m)
    return get_model(mids[0])

def ensure_special_tokens(tok, mdl):
    k = id(tok)
    if _special_tok_ready.get(k):
        return
    _special_tok_ready[k] = True

def protect_codes(s: str):
    repl = {}
    i = [0]
    def sub(m):
        k = f"{PLACE_OPEN}c{i[0]}{PLACE_CLOSE}"
        repl[k] = m.group(0)
        i[0] += 1
        return k
    return CODE2_RE.sub(sub, s), repl

def restore_codes(s: str, repl: dict):
    def sub(m):
        k = f"{PLACE_OPEN}c{m.group(1)}{PLACE_CLOSE}"
        return repl.get(k, m.group(0))
    return RESTORE_RE.sub(sub, s)

def map_m2m(x):
    x = (x or "").lower().strip()
    if "_" in x or "-" in x and len(x) > 2:
        return x
    if x.startswith("en"): return "en"
    if x in {"nb","nob","no","norwegian","bokmål","bokmaal"}: return "no"
    if x in {"nn","nno"}: return "no"
    return x[:2] if len(x) >= 2 else x

def map_nllb(x):
    x = (x or "").strip()
    if "_" in x: return x
    y = x.lower()
    if y.startswith("en"): return "eng_Latn"
    if y in {"nb","nob","no","norwegian","bokmål","bokmaal"}: return "nob_Latn"
    if y in {"nn","nno"}: return "nno_Latn"
    if y in {"fi","fin","finnish"}: return "fin_Latn"
    if y in {"sv","swe","swedish"}: return "swe_Latn"
    if y in {"da","dan","danish"}: return "dan_Latn"
    if y in {"de","deu","ger","german"}: return "deu_Latn"
    if y in {"fr","fra","fre","french"}: return "fra_Latn"
    if y in {"nl","nld","dut","dutch"}: return "nld_Latn"
    if y in {"es","spa","spanish"}: return "spa_Latn"
    if y in {"it","ita","italian"}: return "ita_Latn"
    if y in {"pl","pol","polish"}: return "pol_Latn"
    if y in {"et","est","estonian"}: return "est_Latn"
    if y in {"lt","lit","lithuanian"}: return "lit_Latn"
    if y in {"lv","lvs","lav","latvian"}: return "lvs_Latn"
    if y in {"cs","ces","czech"}: return "ces_Latn"
    if y in {"sk","slk","slovak"}: return "slk_Latn"
    if y in {"ro","ron","rum","romanian"}: return "ron_Latn"
    if y in {"ru","rus","russian"}: return "rus_Cyrl"
    if y in {"uk","ukr","ukrainian"}: return "ukr_Cyrl"
    if y in {"zh","zho","chi","cn"}: return "zho_Hans"
    if y in {"ja","jpn","japanese"}: return "jpn_Jpan"
    if y in {"ko","kor","korean"}: return "kor_Hang"
    if y in {"tr","tur","turkish"}: return "tur_Latn"
    return "eng_Latn"

def model_langs(mid, src_ui, tgt_ui):
    m = (mid or "").lower()
    if "m2m100" in m:
        return map_m2m(src_ui), map_m2m(tgt_ui), "m2m"
    if "helsinki-nlp/opus-mt" in m or "/opus-mt-" in m:
        return src_ui, tgt_ui, "marian"
    return map_nllb(src_ui), map_nllb(tgt_ui), "nllb"

def batch_gen(mid, mdl, tok, texts, src_ui, tgt_ui, max_new=64, beams=4, temperature=0.0, top_p=1.0):
    src_code, tgt_code, kind = model_langs(mid, src_ui, tgt_ui)
    if kind == "marian" and tgt_code not in ("en","eng_Latn"):
        mid, mdl, tok = pick_model_for_target(None, tgt_ui)
        src_code, tgt_code, kind = model_langs(mid, src_ui, tgt_ui)
    if hasattr(tok, "src_lang"):
        tok.src_lang = src_code
    ensure_special_tokens(tok, mdl)
    pre, maps = [], []
    for t in texts:
        s = str(t or "").strip()
        if MASK_CODES:
            a, repl = protect_codes(s)
        else:
            a, repl = s, {}
        pre.append(a)
        maps.append(repl)
    enc = tok(pre, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    bos = None
    if kind == "m2m" and hasattr(tok, "get_lang_id"):
        bos = tok.get_lang_id(map_m2m(tgt_ui))
    elif kind == "nllb":
        try:
            bos = tok.convert_tokens_to_ids(map_nllb(tgt_ui))
        except Exception:
            bos = None
    gen_kwargs = dict(**enc, max_new_tokens=max_new, min_new_tokens=1, length_penalty=1.0, no_repeat_ngram_size=2, early_stopping=True)
    if temperature and float(temperature) > 0:
        gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=float(top_p), num_beams=1))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=max(1, int(beams))))
    if bos is not None:
        gen_kwargs["forced_bos_token_id"] = bos
    with torch.no_grad():
        out_ids = mdl.generate(**gen_kwargs)
    outs = tok.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    post = []
    for o, repl in zip(outs, maps):
        x = o.replace("\u2581", " ")
        x = re.sub(r"(?i)(?:[<\[\(｟]\s*)?c\s*([0-9]{1,3})(?:\s*[>\]\)｠])?", rf"{PLACE_OPEN}c\1{PLACE_CLOSE}", x)
        x = re.sub(r"\s+", " ", x).strip()
        x = restore_codes(x, repl)
        post.append(x)
    return post

def translate_many(texts, src_lang, tgt_lang, batch_size=FAST_BATCH, model_id=None, max_new=64, beams=4, temperature=0.0, top_p=1.0):
    mid, mdl, tok = pick_model_for_target(model_id, tgt_lang)
    todo = list(texts)
    out = [None] * len(todo)
    for j in range(0, len(todo), batch_size):
        chunk = todo[j:j+batch_size]
        res = batch_gen(mid, mdl, tok, chunk, src_lang, tgt_lang, max_new=max_new, beams=beams, temperature=temperature, top_p=top_p)
        for k, r in enumerate(res):
            out[j+k] = r
    return out

def apply_glossary(text, pairs):
    if not pairs:
        out = text
    else:
        out = text
        for src, tgt in pairs:
            if not src or not tgt:
                continue
            pat = re.compile(rf"(?i)(?<!\w){re.escape(src)}(?!\w)")
            out = pat.sub(tgt, out)
    out = re.sub(r"(?i)\b(\d+)\s*grades\b", r"\1 degrees", out)
    return out

def index_html():
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
.container{max-width:980px;margin:40px auto;padding:24px;background:#12161b;border-radius:18px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
h1{font-size:24px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.row{display:flex;gap:12px;align-items:center;margin:12px 0}
.card{background:#0f1317;border:1px solid #1d232b;border-radius:14px;padding:16px}
label{font-size:13px;color:#a7b0bc}
input[type=file],select,button,input[type=text],input[type=number]{width:100%}
select,input[type=file],input[type=text],input[type=number]{background:#0b0f13;border:1px solid #27303b;color:#e8eef5;border-radius:12px;padding:10px;font-size:14px}
button{background:#2f6feb;border:0;color:#fff;border-radius:12px;padding:12px 14px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
small{color:#97a3b2}
hr{border:0;border-top:1px solid #1d232b;margin:16px 0}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#1a2130;border:1px solid #27303b;font-size:12px}
.notice{background:#151b25;border:1px solid #27303b;padding:10px 12px;border-radius:10px;margin:12px 0;font-size:14px}
.progress{height:10px;background:#0b0f13;border:1px solid #27303b;border-radius:999px;overflow:hidden}
.bar{height:100%;width:0%}
footer{opacity:.7;font-size:12px;margin-top:12px}
.help{font-size:12px;opacity:.85;margin-top:6px}
</style>
</head>
<body>
<div id="app" class="container">
  <h1>Excel Column Translator</h1>
  <div class="notice">{{ notice }}</div>
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

      <div class="grid" style="margin-top:12px">
        <div>
          <label>Source language</label>
          <select v-model="srcLang">
            <option value="nob_Latn">Norwegian Bokmål (nob_Latn)</option>
            <option value="nno_Latn">Norwegian Nynorsk (nno_Latn)</option>
            <option value="eng_Latn">English (eng_Latn)</option>
            <option value="fin_Latn">Finnish (fin_Latn)</option>
            <option value="swe_Latn">Swedish (swe_Latn)</option>
            <option value="dan_Latn">Danish (dan_Latn)</option>
            <option value="deu_Latn">German (deu_Latn)</option>
            <option value="fra_Latn">French (fra_Latn)</option>
            <option value="nld_Latn">Dutch (nld_Latn)</option>
            <option value="spa_Latn">Spanish (spa_Latn)</option>
            <option value="ita_Latn">Italian (ita_Latn)</option>
            <option value="pol_Latn">Polish (pol_Latn)</option>
            <option value="est_Latn">Estonian (est_Latn)</option>
            <option value="lit_Latn">Lithuanian (lit_Latn)</option>
            <option value="lvs_Latn">Latvian (lvs_Latn)</option>
            <option value="ces_Latn">Czech (ces_Latn)</option>
            <option value="slk_Latn">Slovak (slk_Latn)</option>
            <option value="ron_Latn">Romanian (ron_Latn)</option>
            <option value="rus_Cyrl">Russian (rus_Cyrl)</option>
            <option value="ukr_Cyrl">Ukrainian (ukr_Cyrl)</option>
            <option value="zho_Hans">Chinese Hans (zho_Hans)</option>
            <option value="jpn_Jpan">Japanese (jpn_Jpan)</option>
            <option value="kor_Hang">Korean (kor_Hang)</option>
            <option value="tur_Latn">Turkish (tur_Latn)</option>
            <option value="en">English m2m (en)</option>
            <option value="no">Norwegian m2m (no)</option>
            <option value="fi">Finnish m2m (fi)</option>
            <option value="sv">Swedish m2m (sv)</option>
            <option value="da">Danish m2m (da)</option>
            <option value="de">German m2m (de)</option>
            <option value="fr">French m2m (fr)</option>
            <option value="nl">Dutch m2m (nl)</option>
            <option value="es">Spanish m2m (es)</option>
            <option value="it">Italian m2m (it)</option>
            <option value="pl">Polish m2m (pl)</option>
          </select>
        </div>
        <div>
          <label>Target language</label>
          <select v-model="tgtLang">
            <option value="eng_Latn">English (eng_Latn)</option>
            <option value="nob_Latn">Norwegian Bokmål (nob_Latn)</option>
            <option value="nno_Latn">Norwegian Nynorsk (nno_Latn)</option>
            <option value="fin_Latn">Finnish (fin_Latn)</option>
            <option value="swe_Latn">Swedish (swe_Latn)</option>
            <option value="dan_Latn">Danish (dan_Latn)</option>
            <option value="deu_Latn">German (deu_Latn)</option>
            <option value="fra_Latn">French (fra_Latn)</option>
            <option value="nld_Latn">Dutch (nld_Latn)</option>
            <option value="spa_Latn">Spanish (spa_Latn)</option>
            <option value="ita_Latn">Italian (ita_Latn)</option>
            <option value="pol_Latn">Polish (pol_Latn)</option>
            <option value="est_Latn">Estonian (est_Latn)</option>
            <option value="lit_Latn">Lithuanian (lit_Latn)</option>
            <option value="lvs_Latn">Latvian (lvs_Latn)</option>
            <option value="ces_Latn">Czech (ces_Latn)</option>
            <option value="slk_Latn">Slovak (slk_Latn)</option>
            <option value="ron_Latn">Romanian (ron_Latn)</option>
            <option value="rus_Cyrl">Russian (rus_Cyrl)</option>
            <option value="ukr_Cyrl">Ukrainian (ukr_Cyrl)</option>
            <option value="zho_Hans">Chinese Hans (zho_Hans)</option>
            <option value="jpn_Jpan">Japanese (jpn_Jpan)</option>
            <option value="kor_Hang">Korean (kor_Hang)</option>
            <option value="tur_Latn">Turkish (tur_Latn)</option>
            <option value="en">English m2m (en)</option>
            <option value="no">Norwegian m2m (no)</option>
            <option value="fi">Finnish m2m (fi)</option>
            <option value="sv">Swedish m2m (sv)</option>
            <option value="da">Danish m2m (da)</option>
            <option value="de">German m2m (de)</option>
            <option value="fr">French m2m (fr)</option>
            <option value="nl">Dutch m2m (nl)</option>
            <option value="es">Spanish m2m (es)</option>
            <option value="it">Italian m2m (it)</option>
            <option value="pl">Polish m2m (pl)</option>
          </select>
        </div>
      </div>

      <div class="grid" style="margin-top:12px">
        <div>
          <label>Model</label>
          <select v-model="model">
            <option v-for="m in models" :key="m" :value="m">{{m}}</option>
          </select>
          <div class="help">Pick which translation model to use.</div>
        </div>
        <div>
          <label>Quality preset</label>
          <select v-model="preset" @change="applyPreset">
            <option value="accurate">Accurate - safe</option>
            <option value="balanced">Balanced</option>
            <option value="creative">Creative</option>
          </select>
          <div class="help">Choose how cautious or free the output should be.</div>
        </div>
      </div>

      <div class="grid" style="margin-top:12px">
        <div>
          <label>Max new tokens</label>
          <input type="number" min="4" max="256" v-model.number="maxNew">
          <div class="help">Upper limit for how many tokens the model adds.</div>
        </div>
        <div>
          <label>Beams</label>
          <input type="number" min="1" max="8" v-model.number="beams">
          <div class="help">Higher is more careful and consistent. 1 is fastest. Set to 1 when Temperature is above 0.</div>
        </div>
      </div>

      <div class="grid" style="margin-top:12px">
        <div>
          <label>Temperature</label>
          <input type="number" step="0.1" min="0" max="2" v-model.number="temperature">
          <div class="help">0 is strict and literal. Higher adds variation. 0.5 to 0.9 gives more creative wording.</div>
        </div>
        <div>
          <label>Top-p</label>
          <input type="number" step="0.05" min="0.1" max="1" v-model.number="topP">
          <div class="help">Limits how adventurous sampling is. 1 keeps all options. 0.9 narrows choices when Temperature is above 0.</div>
        </div>
      </div>

      <hr>

      <div class="grid">
        <div>
          <label>Upload glossary (TSV or CSV, two columns)</label>
          <input type="file" @change="onLex" accept=".tsv,.csv,.txt" />
          <div class="row">
            <button :disabled="!lex || !token" @click="uploadLexicon">Upload glossary</button>
            <div class="badge" v-if="lexCount>0">{{lexCount}} entries</div>
          </div>
        </div>
        <div>
          <label>Glossary mode</label>
          <select v-model="glossaryMode">
            <option value="off">Off</option>
            <option value="replace">Replace after MT</option>
          </select>
          <div class="help">Optional replacements applied after translation. Use for fixed terminology.</div>
        </div>
      </div>

      <hr>
      <div class="row" v-if="jobId">
        <div class="progress" style="flex:1"><div class="bar" :style="{width: progressPct+'%', background: '#2f6feb'}"></div></div>
        <div class="badge">{{stage}} {{done}}/{{total}}</div>
      </div>
      <div class="notice help" v-if="jobId">{{ explain }}</div>
      <div class="row">
        <button :disabled="!readyToTranslate || loading || jobId" @click="start">Translate</button>
        <div class="badge" v-if="status">{{status}}</div>
      </div>
      <small>Pure MT with code preservation</small>
    </div>
  </div>
  <footer>{{ notice }}</footer>
</div>
<script src="https://unpkg.com/vue@3"></script>
<script>
const app = Vue.createApp({
  data(){return{
    file:null,token:null,sheets:[],sheet:null,columns:[],srcCol:null,tgtCol:null,mode:"append_new",
    srcLang:"nob_Latn",tgtLang:"eng_Latn",status:"",loading:false,jobId:null,progressPct:0,done:0,total:0,stage:"",
    models:[],model:null,maxNew:64,beams:4,temperature:0,topP:1.0,
    preset:"accurate",
    lex:null,lexCount:0,glossaryMode:"off"
  }},
  computed:{
    readyToTranslate(){return this.token&&this.sheet&&this.srcCol&&this.tgtCol},
    notice(){const m=this.models&&this.models.length?this.models.join(", "):(this.model||"");return m||"No models configured"},
    explain(){
      if(!this.jobId) return "";
      if(this.stage==="translating") return "Working through unique values."
      if(this.stage==="writing") return "Writing results to the workbook."
      if(this.stage==="done") return "Finished."
      return ""
    }
  },
  methods:{
    applyPreset(){
      if(this.preset==="accurate"){this.beams=6;this.temperature=0;this.topP=1.0;if(this.maxNew<64)this.maxNew=64}
      if(this.preset==="balanced"){this.beams=4;this.temperature=0.3;this.topP=0.9;if(this.maxNew<64)this.maxNew=64}
      if(this.preset==="creative"){this.beams=1;this.temperature=0.8;this.topP=0.9;if(this.maxNew<128)this.maxNew=128}
    },
    async refreshStatus(){try{const r=await fetch("/status");if(!r.ok)return;const j=await r.json();this.models=j.models||[];this.model=j.model||this.models[0]||null}catch(e){}},
    onFile(e){this.file=e.target.files[0];this.token=null;this.sheets=[];this.columns=[];this.srcCol=null;this.tgtCol=null;this.lex=null;this.lexCount=0},
    onLex(e){this.lex=e.target.files[0]},
    async uploadLexicon(){
      if(!this.lex||!this.token)return;
      const fd=new FormData();
      fd.append("token",this.token);
      fd.append("file",this.lex);
      const r=await fetch("/upload_lexicon",{method:"POST",body:fd});
      if(!r.ok)return;
      const j=await r.json();
      this.lexCount=j.count||0
    },
    async inspect(){
      if(!this.file)return;
      this.loading=true;this.status="Inspecting";
      const fd=new FormData();fd.append("file",this.file);
      const r=await fetch("/inspect",{method:"POST",body:fd});
      if(!r.ok){this.status="Failed to read file";this.loading=false;return}
      const j=await r.json();
      this.token=j.token;this.sheets=j.sheets;this.sheet=j.sheets[0]||null;this.columns=j.columns||[];
      this.srcCol=this.columns[0]||null;this.tgtCol=this.srcCol?this.srcCol+"_tr":null;this.status="Ready";this.loading=false
    },
    async fetchColumns(){
      if(!this.token||!this.sheet)return;
      this.loading=true;this.status="Loading columns";
      const fd=new FormData();fd.append("token",this.token);fd.append("sheet",this.sheet);
      const r=await fetch("/columns",{method:"POST",body:fd});
      if(!r.ok){this.status="Failed to load columns";this.loading=false;return}
      const j=await r.json();this.columns=j.columns||[];if(!this.srcCol)this.srcCol=this.columns[0]||null;this.status="Ready";this.loading=false
    },
    async start(){
      this.status="Starting";
      const fd=new FormData();
      fd.append("token",this.token);fd.append("sheet",this.sheet);
      fd.append("src_col",this.srcCol);fd.append("tgt_col",this.tgtCol);
      fd.append("src_lang",this.srcLang);fd.append("tgt_lang",this.tgtLang);
      fd.append("mode",this.mode);fd.append("model",this.model||"");
      fd.append("max_new",String(this.maxNew));fd.append("beams",String(this.beams));
      fd.append("temperature",String(this.temperature));fd.append("top_p",String(this.topP));
      fd.append("glossary_mode",this.glossaryMode);
      const r=await fetch("/start",{method:"POST",body:fd});
      if(!r.ok){this.status="Failed to start";return}
      const j=await r.json();this.jobId=j.job;this.status="Translating";this.poll()
    },
    async poll(){
      if(!this.jobId)return;
      const r=await fetch(`/job?job=${this.jobId}`);
      if(!r.ok){this.status="Job error";return}
      const j=await r.json();
      this.stage=j.stage;this.done=j.done;this.total=j.total;
      this.progressPct=j.total?Math.min(100,Math.round(100*j.done/j.total)):(j.stage==="done"?100:0);
      if(j.stage==="done"){
        const d=await fetch(`/download?job=${this.jobId}`);
        const blob=await d.blob();const url=URL.createObjectURL(blob);
        const a=document.createElement("a");a.href=url;a.download="translated.xlsx";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
        this.status="Downloaded";this.jobId=null;return
      }
      if(j.stage==="error"){this.status=j.error||"Error";this.jobId=null;return}
      setTimeout(this.poll,500)
    }
  },
  mounted(){this.refreshStatus();this.applyPreset()}
})
app.mount("#app")
</script>
</body>
</html>
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    _ready = True
    def _warm():
        try:
            mid, mdl, tok = get_model()
            batch_gen(mid, mdl, tok, ["hei"], "nob_Latn", "eng_Latn", max_new=4)
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def index():
    return index_html()

@app.get("/status")
def status():
    mids = get_model_ids()
    return {"ready": _ready, "models": mids, "model": mids[0] if mids else None}

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

@app.post("/upload_lexicon")
async def upload_lexicon(token: str = Form(...), file: UploadFile = File(...)):
    if token not in _files:
        return JSONResponse({"error": "Invalid token"}, status_code=400)
    try:
        raw = await file.read()
        txt = raw.decode("utf-8-sig", errors="ignore")
        pairs = []
        for line in txt.splitlines():
            if not line.strip():
                continue
            if "\t" in line:
                a, b = line.rstrip("\n").split("\t", 1)
            elif "," in line:
                a, b = line.rstrip("\n").split(",", 1)
            else:
                continue
            a = a.strip()
            b = b.strip()
            if a and b:
                pairs.append((a, b))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        _glossaries[token] = pairs
        return {"ok": True, "count": len(pairs)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

def run_job(job, token, sheet, src_col, tgt_col, src_lang, tgt_lang, mode, model_id, max_new, beams, temperature, top_p, glossary_mode):
    try:
        _jobs[job]["stage"] = "loading"
        xf = pd.ExcelFile(io.BytesIO(_files[token]))
        df = xf.parse(sheet)
        if src_col not in df.columns:
            _jobs[job]["stage"] = "error"; _jobs[job]["error"] = "Missing source column"; return
        if tgt_col not in df.columns:
            df[tgt_col] = ""
        else:
            df[tgt_col] = df[tgt_col].fillna("")
        cache = load_cache()
        src_vals = df[src_col].fillna("").astype(str)
        tgt_vals = df[tgt_col].fillna("").astype(str)
        todo_idx, todo_texts = [], []
        for i, (s, t) in enumerate(zip(src_vals, tgt_vals)):
            s2 = s.strip()
            if s2 == "" or s2.startswith("="):
                continue
            if mode == "skip_filled" and t.strip() != "":
                continue
            if mode == "append_new" and t.strip() != "":
                continue
            key = f"{CACHE_VER}|{src_lang}|{tgt_lang}|{s2}"
            if key not in cache:
                todo_texts.append(s2)
            todo_idx.append(i)
        uniq = list(dict.fromkeys(todo_texts))
        _jobs[job]["total"] = max(len(uniq), 1)
        pairs = _glossaries.get(token) if glossary_mode == "replace" else None
        if uniq:
            _jobs[job]["stage"] = "translating"
            outs = translate_many(uniq, src_lang, tgt_lang, model_id=model_id, max_new=max_new, beams=beams, temperature=temperature, top_p=top_p)
            for i, s in enumerate(uniq, 1):
                y = outs[i-1]
                if pairs:
                    y = apply_glossary(y, pairs)
                cache[f"{CACHE_VER}|{src_lang}|{tgt_lang}|{s}"] = y
                _jobs[job]["done"] = i
            save_cache(cache)
        _jobs[job]["stage"] = "writing"
        for i in todo_idx:
            s2 = src_vals.iat[i].strip()
            df.at[i, tgt_col] = cache.get(f"{CACHE_VER}|{src_lang}|{tgt_lang}|{s2}", "")
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
async def start(
    token: str = Form(...),
    sheet: str = Form(...),
    src_col: str = Form(...),
    tgt_col: str = Form(...),
    src_lang: str = Form(...),
    tgt_lang: str = Form(...),
    mode: str = Form("append_new"),
    model: str = Form(None),
    max_new: int = Form(64),
    beams: int = Form(4),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    glossary_mode: str = Form("off")
):
    if token not in _files:
        return JSONResponse({"error": "Invalid token"}, status_code=400)
    j = str(uuid.uuid4())
    _jobs[j] = {"stage": "queued", "done": 0, "total": 0}
    t = threading.Thread(target=run_job, args=(j, token, sheet, src_col, tgt_col, src_lang, tgt_lang, mode, model, max_new, beams, temperature, top_p, glossary_mode), daemon=True)
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
