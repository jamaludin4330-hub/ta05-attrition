import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
app = Flask(__name__)

bundle = joblib.load(BASE / "model" / "attrition_model.joblib")
MODEL, COLUMNS = bundle["model"], bundle["columns"]
METRICS = json.loads((BASE / "model" / "metrics.json").read_text(encoding="utf-8"))
CAT_COLS = ["OverTime", "MaritalStatus", "BusinessTravel"]

SCALE = [("1", "1 - Rendah"), ("2", "2 - Sedang"), ("3", "3 - Tinggi"), ("4", "4 - Sangat tinggi")]

# Definisi form: satu sumber untuk tampilan, validasi, dan nama fitur.
FIELDS = [
    {"group": "Profil Karyawan", "icon": "👤", "items": [
        {"name": "Age", "label": "Usia (tahun)", "kind": "number", "min": 18, "max": 60, "default": 30},
        {"name": "MaritalStatus", "label": "Status pernikahan", "kind": "select", "default": "Married",
         "options": [("Single", "Lajang"), ("Married", "Menikah"), ("Divorced", "Cerai")]},
        {"name": "DistanceFromHome", "label": "Jarak rumah ke kantor (km)", "kind": "number", "min": 1, "max": 29, "default": 8},
    ]},
    {"group": "Pekerjaan & Kompensasi", "icon": "💼", "items": [
        {"name": "MonthlyIncome", "label": "Pendapatan bulanan", "kind": "number", "min": 1000, "max": 20000,
         "default": 5000, "hint": "Satuan sesuai dataset (1.009 - 19.999)"},
        {"name": "JobLevel", "label": "Level jabatan", "kind": "select", "default": "2",
         "options": [("1", "1 - Junior"), ("2", "2 - Staf"), ("3", "3 - Senior"), ("4", "4 - Manajer"), ("5", "5 - Direktur")]},
        {"name": "StockOptionLevel", "label": "Level opsi saham", "kind": "select", "default": "0",
         "options": [("0", "0 - Tidak ada"), ("1", "1"), ("2", "2"), ("3", "3")]},
        {"name": "OverTime", "label": "Bekerja lembur?", "kind": "select", "default": "No",
         "options": [("No", "Tidak"), ("Yes", "Ya")]},
        {"name": "BusinessTravel", "label": "Frekuensi perjalanan dinas", "kind": "select", "default": "Travel_Rarely",
         "options": [("Non-Travel", "Tidak pernah"), ("Travel_Rarely", "Jarang"), ("Travel_Frequently", "Sering")]},
    ]},
    {"group": "Riwayat Karier", "icon": "📈", "items": [
        {"name": "TotalWorkingYears", "label": "Total pengalaman kerja (tahun)", "kind": "number", "min": 0, "max": 40, "default": 8},
        {"name": "YearsAtCompany", "label": "Lama di perusahaan ini (tahun)", "kind": "number", "min": 0, "max": 40, "default": 3},
        {"name": "NumCompaniesWorked", "label": "Jumlah perusahaan sebelumnya", "kind": "number", "min": 0, "max": 9, "default": 2},
    ]},
    {"group": "Kepuasan & Keseimbangan", "icon": "😊", "items": [
        {"name": "JobSatisfaction", "label": "Kepuasan terhadap pekerjaan", "kind": "select", "default": "3", "options": SCALE},
        {"name": "EnvironmentSatisfaction", "label": "Kepuasan terhadap lingkungan kerja", "kind": "select", "default": "3", "options": SCALE},
        {"name": "WorkLifeBalance", "label": "Keseimbangan kerja dan hidup", "kind": "select", "default": "3", "options": SCALE},
    ]},
]
ALL_ITEMS = [i for g in FIELDS for i in g["items"]]

NICE = {i["name"]: i["label"].split(" (")[0].rstrip("?") for i in ALL_ITEMS}
NICE["OverTime"] = "Lembur"
CAT_TEXT = {(i["name"], o[0]): o[1] for i in ALL_ITEMS if i["kind"] == "select" for o in i["options"]}


def default_values():
    return {it["name"]: str(it["default"]) for it in ALL_ITEMS}


def parse_form(form):
    values, errors, clean = {}, {}, {}
    for it in ALL_ITEMS:
        name = it["name"]
        raw = (form.get(name) or "").strip()
        values[name] = raw
        if it["kind"] == "number":
            try:
                num = float(raw)
            except ValueError:
                errors[name] = "Isi dengan angka."
                continue
            if not it["min"] <= num <= it["max"]:
                errors[name] = f"Nilai harus antara {it['min']} dan {it['max']}."
                continue
            clean[name] = num
        else:
            if raw not in [o[0] for o in it["options"]]:
                errors[name] = "Pilih salah satu opsi."
                continue
            clean[name] = raw if name in CAT_COLS else float(raw)
    return values, errors, clean


def build_row(clean):
    df = pd.DataFrame([clean])
    df = pd.get_dummies(df, columns=CAT_COLS, dtype=int)
    return df.reindex(columns=COLUMNS, fill_value=0)


def decision_path(row):
    """Terjemahkan jalur dari akar ke daun menjadi kalimat yang bisa dibaca."""
    tree = MODEL.tree_
    steps = []
    for nid in MODEL.decision_path(row).indices:
        if tree.children_left[nid] == tree.children_right[nid]:
            continue
        feat = COLUMNS[tree.feature[nid]]
        thr, val = tree.threshold[nid], float(row.iloc[0, tree.feature[nid]])
        base = next((c for c in CAT_COLS if feat.startswith(c + "_")), None)
        if base:
            cat = feat[len(base) + 1:]
            teks = CAT_TEXT.get((base, cat), cat)
            if base == "OverTime":
                ya = (cat == "Yes") == (val > thr)
                steps.append(f"Lembur: {'Ya' if ya else 'Tidak'}")
            else:
                steps.append(f"{NICE[base]}: {teks}" if val > thr else f"{NICE[base]}: bukan {teks}")
        else:
            steps.append(f"{NICE[feat]} = {val:g} ({'≤' if val <= thr else '>'} {thr:.1f})")
    return steps


@app.route("/")
def home():
    return render_template("index.html", m=METRICS)


@app.route("/prediksi", methods=["GET", "POST"])
def predict():
    values, errors, result = default_values(), {}, None
    if request.method == "POST":
        values, errors, clean = parse_form(request.form)
        if not errors:
            row = build_row(clean)
            proba = float(MODEL.predict_proba(row)[0][1])
            resign = int(MODEL.predict(row)[0]) == 1
            result = {"resign": resign, "proba": round(proba * 100, 1),
                      "path": decision_path(row)}
    return render_template("predict.html", fields=FIELDS, values=values, errors=errors, result=result)


@app.route("/analisis")
def analysis():
    return render_template("analysis.html", m=METRICS)


@app.route("/tentang")
def about():
    return render_template("about.html", m=METRICS)


@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)