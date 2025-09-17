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
python -m uvicorn app:app --reload
