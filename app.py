import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json, time
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="PSEW v3", layout="wide")

APP_DIR = Path(__file__).parent

# ---------- i18n ----------
I18N = json.loads(Path(APP_DIR/"i18n.json").read_text(encoding="utf-8"))
LANGS = {"fr":"Français", "bm":"Bamanankan"}

def t(key):
    lang = st.session_state.get("lang","fr")
    return I18N.get(lang, {}).get(key, key)

if "lang" not in st.session_state:
    st.session_state.lang = "fr"
if "mobile" not in st.session_state:
    st.session_state.mobile = False

# ---------- Load config ----------
CFG = json.loads(Path(APP_DIR/"config.json").read_text(encoding="utf-8"))

# ---------- State ----------
if "sites" not in st.session_state:
    st.session_state.sites = {}
    for s in CFG["sites"]:
        st.session_state.sites[s["id"]] = {
            "id":s["id"],"name":s["name"],"lat":s["lat"],"lon":s["lon"],"pop":s.get("pop",0.0),
            "R":0.3,"latchedRed":False,"lastRAG":"GREEN",
            "noiseRadar":1.0,"noiseRF":1.0,"noiseTraffic":1.0,"noiseAcoustic":1.0,
            "incident":False,"updates":0,"scans":0
        }
if "running" not in st.session_state: st.session_state.running=False
if "audit" not in st.session_state: st.session_state.audit=[]
if "metrics" not in st.session_state:
    st.session_state.metrics={"truthEvents":0,"detectedEvents":0,"falseAlarms":0,"emptyCells":0,
                              "Pd_hist":[], "Pfa_hist":[], "Cont_hist":[]}

# ---------- Sidebar ----------
with st.sidebar:
    st.title(t("controls"))
    st.selectbox(t("language"), options=list(LANGS.keys()), format_func=lambda k: LANGS[k],
                 index=["fr","bm"].index(st.session_state.lang) if st.session_state.lang in ["fr","bm"] else 0,
                 key="lang")
    st.toggle(t("mobile_mode"), key="mobile")

    st.caption(t("thresholds"))
    theta_high = st.slider(t("theta_high"), 1.5, 4.0, float(CFG["thresholds"]["theta_high"]), 0.1)
    theta_low  = st.slider(t("theta_low"),  0.5, 3.0, float(CFG["thresholds"]["theta_low"]), 0.1)

    st.caption(t("weights"))
    w_radar   = st.slider(t("radar_weight"),   0.0, 0.6, float(CFG["weights"]["radar"]), 0.05)
    w_rf      = st.slider(t("rf_weight"),      0.0, 0.6, float(CFG["weights"]["rf"]), 0.05)
    w_traffic = st.slider(t("traffic_weight"), 0.0, 0.6, float(CFG["weights"]["traffic"]), 0.05)
    w_ac      = st.slider(t("acoustic_weight"),0.0, 0.6, float(CFG["weights"]["acoustic"]), 0.05)

    st.caption(t("ethics"))
    require_multi = st.checkbox(t("require_multi"), value=bool(CFG["ethics"]["require_multi_confirm"]))
    require_human = st.checkbox(t("require_human"), value=bool(CFG["ethics"]["require_human_auth"]))
    data_min      = st.checkbox(t("data_min"),      value=bool(CFG["ethics"]["data_minimization"]))

    st.caption(t("layers"))
    show_zones     = st.checkbox(t("zones"), value=True)
    show_corridors = st.checkbox(t("corridors"), value=True)
    show_assets    = st.checkbox(t("assets"), value=True)

    st.caption(t("simulation"))
    colA, colB, colC, colD = st.columns(4)
    if colA.button(t("run_pause") if not st.session_state.running else t("pause")):
        st.session_state.running = not st.session_state.running
    if colB.button(t("step")):
        st.session_state.running = False
        st.session_state._step_once = True
    if colC.button(t("incident")):
        import random, time as _t
        key = random.choice(list(st.session_state.sites.keys()))
        st.session_state.sites[key]["incident"] = True
        st.session_state.sites[key]["incident_until"] = _t.time() + 8.0
    if colD.button(t("reset")):
        st.session_state.clear()
        st.experimental_rerun()

# ---------- CSS for mobile ----------
if st.session_state.mobile:
    st.markdown("""
        <style>
          .block-container {padding-top:0.5rem; padding-bottom:0.5rem; max-width: 880px;}
          h1,h2,h3 {font-size: 1.0rem !important;}
          .stMetric {font-size: 0.8rem !important;}
        </style>
    """, unsafe_allow_html=True)

def clamp(x,a,b): 
    return max(a, min(b,x))

def rag_color_rgb(r):
    return [220,38,38,180] if r=="RED" else [245,158,11,180] if r=="AMBER" else [22,163,74,180]

def circle_polygon(lat, lon, radius_km=15, steps=48):
    pts=[]; import numpy as np
    for i in range(steps):
        ang = 2*np.pi * i/steps
        dlat = (radius_km/111.0) * np.sin(ang)
        dlon = (radius_km/(111.0*np.cos(np.radians(lat)))) * np.cos(ang)
        pts.append([lon+dlon, lat+dlat])
    pts.append(pts[0])
    return [pts]

def tick():
    import numpy as np, time as _t, random
    S = st.session_state.sites
    M = st.session_state.metrics
    if np.random.rand() < CFG.get("simulation", {}).get("incident_prob_per_tick", 0.08):
        key = random.choice(list(S.keys()))
        S[key]["incident"] = True
        S[key]["incident_until"] = _t.time() + CFG.get("simulation", {}).get("incident_duration_s", 10.0)
    top_rag = "GREEN"
    for key, stt in S.items():
        if stt.get("incident_until") and _t.time() > stt["incident_until"]:
            stt["incident"] = False
        # sensor draws
        def norm_like(): return (np.random.rand()*2-1)*0.7
        rad = 0.2 + norm_like(); rf = 0.2 + norm_like(); traf=0.2+norm_like(); ac=0.2+norm_like()
        if stt["incident"]:
            rad += 2.0 + np.random.rand()*0.8
            rf  += 1.6 + np.random.rand()*0.8
            ac  += 1.4 + np.random.rand()*0.6
            traf+= 1.2 + np.random.rand()*0.6
        normRad  = rad  / max(0.6, stt["noiseRadar"])
        normRF   = rf   / max(0.6, stt["noiseRF"])
        normTraf = traf / max(0.6, stt["noiseTraffic"])
        normAc   = ac   / max(0.6, stt["noiseAcoustic"])
        lam = CFG.get("normalization", {}).get("ewma_lambda", 0.05)
        stt["noiseRadar"]    = (1-lam)*stt["noiseRadar"]    + lam*max(0.6, rad)
        stt["noiseRF"]       = (1-lam)*stt["noiseRF"]       + lam*max(0.6, rf)
        stt["noiseTraffic"]  = (1-lam)*stt["noiseTraffic"]  + lam*max(0.6, traf)
        stt["noiseAcoustic"] = (1-lam)*stt["noiseAcoustic"] + lam*max(0.6, ac)
        benign = 0.2 + 0.3*np.random.rand()
        Rnext = (1-0.15)*stt["R"] + 0.15*(w_radar*normRad + w_rf*normRF + w_traffic*normTraf + w_ac*normAc - 0.6*benign)
        stt["R"] = clamp(Rnext, 0, 6)
        channels_above = sum(v >= CFG.get("decision", {}).get("confirm_level", 1.5) for v in [normRad, normRF, normTraf, normAc])
        require_ok = (not require_multi) or (channels_above >= CFG.get("decision", {}).get("min_channels_for_red", 2))
        rag = "GREEN"
        if stt["latchedRed"]:
            if stt["R"] < theta_low:
                stt["latchedRed"] = False
                rag = "AMBER" if stt["R"] >= theta_low else "GREEN"
            else:
                rag = "RED"
        else:
            if stt["R"] >= theta_high and require_ok:
                rag = "RED"; stt["latchedRed"]=True
            elif stt["R"] >= theta_low: rag = "AMBER"
            else: rag = "GREEN"
        stt["scans"] += 1
        if rag != "GREEN": stt["updates"] += 1
        if stt["incident"]:
            M["truthEvents"] += 1
            if rag=="RED": M["detectedEvents"] += 1
        else:
            M["emptyCells"] += 1
            if rag=="RED": M["falseAlarms"] += 1
        if rag=="RED": top_rag="RED"
        elif rag=="AMBER" and top_rag!="RED": top_rag="AMBER"
        if rag != stt["lastRAG"]:
            sensors_txt = f"radar:{normRad:.2f}, rf:{normRF:.2f}, traffic:{normTraf:.2f}, acoustic:{normAc:.2f}"
            status = "Awaiting human authorization" if (rag=="RED" and require_human) else "Logged"
            st.session_state.audit.insert(0, {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "site": stt["name"],
                "change": f"{stt['lastRAG']} → {rag}",
                "rag": rag, "sensors": sensors_txt,
                "action": ("Verify with second sensor; notify liaison; perimeter safety + medical standby"
                           if rag=="RED" else "Elevate awareness; cross-check with another sensor" if rag=="AMBER" else "Routine monitoring"),
                "status": status
            })
            stt["lastRAG"] = rag
    Pd  = (st.session_state.metrics["detectedEvents"]/st.session_state.metrics["truthEvents"]) if st.session_state.metrics["truthEvents"]>0 else 1.0
    Pfa = (st.session_state.metrics["falseAlarms"]/st.session_state.metrics["emptyCells"]) if st.session_state.metrics["emptyCells"]>0 else 0.0
    cont = np.mean([ (s["updates"]/s["scans"]) if s["scans"]>0 else 0.0 for s in st.session_state.sites.values() ])
    for key,val in [("Pd_hist",Pd),("Pfa_hist",min(Pfa*10.0,1.0)),("Cont_hist",cont)]:
        st.session_state.metrics[key].append(val)
        if len(st.session_state.metrics[key])> (180 if st.session_state.mobile else 360):
            st.session_state.metrics[key] = st.session_state.metrics[key][- (180 if st.session_state.mobile else 360):]
    return top_rag

# ---------- Header ----------
st.markdown(f"### {t('app_title')}")
st.caption(t("tagline"))

# ---------- Main ----------
if st.session_state.get("_step_once"):
    top_rag = tick(); st.session_state._step_once=False
elif st.session_state.running:
    top_rag = tick(); time.sleep(1); st.rerun()
else:
    top_rag = "GREEN"

# Layout adapts in mobile mode
if st.session_state.mobile:
    col_map = st.container()
    col_metrics = st.container()
else:
    col_map, col_metrics = st.columns([1.05,1.25], gap="large")

# --- Map
with col_map:
    df_sites = pd.DataFrame([{
        "lat": s["lat"], "lon": s["lon"], "name": s["name"], "rag": s["lastRAG"],
        "color": rag_color_rgb(s["lastRAG"]), "R": round(s["R"],2)
    } for s in st.session_state.sites.values()])

    layers = [pdk.Layer("ScatterplotLayer", df_sites,
                        get_position='[lon, lat]', get_radius=6000 if st.session_state.mobile else 10000,
                        get_fill_color="color", pickable=True)]
    # Zones
    if show_zones:
        zone_rows = []
        for s in CFG["sites"]:
            rkm = s.get("zone_radius_km", 20)
            zone_rows.append({"name": s["name"], "polygons": circle_polygon(s["lat"], s["lon"], rkm)})
        df_z = pd.DataFrame(zone_rows)
        layers.append(pdk.Layer("PolygonLayer", df_z, get_polygon="polygons",
                                get_fill_color=[100,149,237,30], get_line_color=[100,149,237], line_width_min_pixels=1))
    # Corridors
    if show_corridors and CFG.get("corridors"):
        df_paths = pd.DataFrame([{"name": c["name"], "path": c["path"]} for c in CFG["corridors"]])
        layers.append(pdk.Layer("PathLayer", df_paths, get_path="path", get_color=[255,140,0,180],
                                width_scale=20, width_min_pixels=2, pickable=True))
    # Assets
    if show_assets and CFG.get("assets"):
        df_assets = pd.DataFrame(CFG["assets"]); df_assets["color"] = [[59,130,246,200]]*len(df_assets)
        layers.append(pdk.Layer("ScatterplotLayer", df_assets, get_position='[lon, lat]', get_radius=4000 if st.session_state.mobile else 6000,
                                get_fill_color="color", pickable=True))

    view_state = pdk.ViewState(latitude=17.5, longitude=-3.5, zoom= (4.3 if st.session_state.mobile else 4.5))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None,
                    tooltip={"text":"{name}"}), use_container_width=True)

    st.markdown(f"**{t('sites_label')}:** {', '.join([s['name'] for s in CFG['sites']])}  •  **{t('top_rag')}:** {top_rag}")

    if top_rag == "RED":
        st.error(t("playbook_red"))
    elif top_rag == "AMBER":
        st.warning(t("playbook_amber"))
    else:
        st.info(t("playbook_green"))

# --- Metrics & Audit
with col_metrics:
    st.subheader(t("metrics_title"))
    # Last values
    M = st.session_state.metrics
    Pd_last  = M["Pd_hist"][-1] if M["Pd_hist"] else None
    Pfa_last = M["Pfa_hist"][-1] if M["Pfa_hist"] else None
    Ct_last  = M["Cont_hist"][-1] if M["Cont_hist"] else None
    mcols = st.columns(3) if not st.session_state.mobile else [st.container(), st.container(), st.container()]
    mcols[0].metric(t("pd"), f"{Pd_last*100:.1f}%" if Pd_last is not None else "—")
    mcols[1].metric(t("pfa"), f"{Pfa_last*100:.2f}%" if Pfa_last is not None else "—")
    mcols[2].metric(t("continuity"), f"{Ct_last*100:.1f}%" if Ct_last is not None else "—")

    chart_h = 110 if st.session_state.mobile else 140
    c1, c2, c3 = (st.container(), st.container(), st.container()) if st.session_state.mobile else st.columns(3)
    if M["Pd_hist"]:
        c1.line_chart(M["Pd_hist"], height=chart_h)
        c2.line_chart(M["Pfa_hist"], height=chart_h)
        c3.line_chart(M["Cont_hist"], height=chart_h)
    else:
        st.caption("—")

    st.divider()
    st.subheader(t("audit_log"))
    import pandas as pd
    df_audit = pd.DataFrame(st.session_state.audit)
    if df_audit.empty:
        st.caption(t("no_changes"))
    else:
        st.dataframe(df_audit, use_container_width=True, height=(200 if st.session_state.mobile else 260))
        csv = df_audit.to_csv(index=False).encode("utf-8")
        st.download_button(t("export_csv"), data=csv, file_name="psew_audit_log.csv", mime="text/csv")
