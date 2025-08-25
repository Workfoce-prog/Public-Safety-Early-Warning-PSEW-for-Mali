# PSEW — Full Bundle

Generated: 2025-08-25 21:07

## Contents
- streamlit/psew_streamlit_v3.zip (OK)
- streamlit/psew_streamlit_v2.zip (OK)
- streamlit/psew_streamlit.zip (OK)
- deploy/psew_streamlit_deploy.zip (missing)
- onepagers/PSEW_Leadership_OnePager.html (OK)
- onepagers/PSEW_Leadership_OnePager_multi.html (OK)
- web_demo/psew_dashboard.html (OK)

## How to start (Streamlit v3)
```bash
unzip streamlit/psew_streamlit_v3.zip -d psew_v3
cd psew_v3
python -m venv .venv
# Windows: .\.venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- v3 includes French + Bambara + Mobile Mode.
- v2 includes Oversight & SOPs tab, config upload/download, more layers.
- `deploy/` contains Docker/Compose scaffolding (replace placeholders with v2 files).
- `onepagers/` has leadership briefs (single‑page HTML).
- `web_demo/` is the standalone HTML prototype (no server needed).