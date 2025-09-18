# Excel Column Translator

Translate a column in an Excel file from Norwegian Bokmål or Nynorsk to English, with:
- Offline neural MT using facebook/nllb-200-distilled-600M
- Wiktionary English translations via Wiktextract as a high-precision dictionary
- Dynamic compound splitter for Norwegian compounds like FESTEBØYLEHALVDEL
- Progress bar, caching of repeated strings, unit and degree normalization
- No external paid APIs

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Disk space: about 4 GB free for the model cache
- Packages:
  - fastapi
  - uvicorn[standard]
  - pandas
  - openpyxl
  - transformers
  - torch
  - sentencepiece
  - requests

## Quick start

```powershell
# Windows PowerShell
git clone https://github.com/<yourname>/excel-translator.git
cd excel-translator
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pandas openpyxl transformers sentencepiece requests
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install python-multipart
python -m uvicorn app:app --reload

Open http://127.0.0.1:8000

Upload an .xlsx, choose the sheet and source column, set a target column, keep “Use dictionary first” on, click Translate. The file will download when complete.

First run

The app downloads the Wiktextract dump for Norwegian Bokmål and builds dict/nb_idx.json and dict/nb_idx_ascii.json

The NLLB model is fetched to your Hugging Face cache

You can prefetch the model to show progress in the terminal:

python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; m='facebook/nllb-200-distilled-600M'; AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m); print('cached')"

Useful endpoints
Invoke-RestMethod http://127.0.0.1:8000/status | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:8000/clear_cache
# URL encode non ASCII when testing
$term=[uri]::EscapeDataString('FESTEBØYLEHALVDEL'); Invoke-RestMethod "http://127.0.0.1:8000/dict_test?term=$term" | ConvertTo-Json

Behavior

Dictionary is used first for single words

For unknown single words the splitter runs a DP search over the ASCII folded form to find parts found in the dictionary, only accepting splits that map every part to a known dictionary entry

If dictionary paths fail, neural MT is used

All caps in source yields all caps output

Units and degree symbols are normalized: mm, No., 90°

Codes like CU125, RN7, 0880110.B0069 are preserved

Configuration

Environment variables:

HF_HOME Hugging Face cache directory

MODEL_ROOT local path if you want to store the model under models/facebook/nllb-200-distilled-600M

DICT_DIR directory to store the Wiktextract index

MT_CACHE_PATH path to the app translation cache

Example:

$env:HF_HOME = "C:\hf-cache"
$env:DICT_DIR = ".\dict"
python -m uvicorn app:app --reload

GitHub usage

Add these files at repo root:

app.py

.gitignore

requirements.txt

README.md

requirements.txt
fastapi
uvicorn[standard]
pandas
openpyxl
transformers
torch
sentencepiece
requests

.gitignore
.venv/
__pycache__/
*.pyc
dict/kaikki.org-dictionary-NorwegianBokmål.jsonl
dict/nb_idx.json
dict/nb_idx_ascii.json
.translate_cache.json
models/
*.xlsx

Local install from Git
git clone https://github.com/<yourname>/excel-translator.git
cd excel-translator
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --reload

Troubleshooting

If /status shows downloading for a long time, prefetch the model as shown above

On Windows, enabling Developer Mode improves cache performance for symlinks

If translations look stale, clear the app cache:

Invoke-RestMethod -Method Post http://127.0.0.1:8000/clear_cache


If PowerShell mangles letters like Ø in test URLs, URL encode with [uri]::EscapeDataString(...)

Notes

This app works offline after the first run

Accuracy improves on domain terms because compounds are split to known parts before MT


If FESTEBØYLEHALVDEL still shows as one piece in `dict_test`, run:
```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/clear_cache


then retry the test with the URL encoded term. The DP splitter will accept only splits that map every part to a dictionary entry, which yields something like attachment bracket half.
