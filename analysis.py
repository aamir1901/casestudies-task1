"""
Case Studies in Data Science - Individual Task 1, Part 1.3
At-risk student classification on OULAD and Student Performance (Cortez)
Models: Random Forest (ensemble) and Multilayer Perceptron neural network.

Evaluation approach (recall-priority for the at-risk class, F1, ROC-AUC,
confusion matrices) based on Page (2007), Evaluating Machine Learning
Methods, CS 760 UW-Madison, slides 18-29:
http://pages.cs.wisc.edu/~dpage/cs760/evaluating.pdf

Datasets:
- OULAD: Kuzilek, Hlosta & Zdrahal (2017), Scientific Data 4:170171.
  https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Student Performance: Cortez & Silva (2008).
  https://archive.ics.uci.edu/dataset/320/student+performance

Implementation uses scikit-learn (Pedregosa et al., 2011, JMLR 12).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = 42
UP = "/mnt/user-data/uploads"


# ----------------------------------------------------------------------
# 1. OULAD: student-level feature engineering
# ----------------------------------------------------------------------
def build_oulad():
    info = pd.read_csv(f"{UP}/studentInfo.csv")

    # Aggregate the 10.65M-row VLE interaction log to one row per
    # student-module-presentation: total engagement, EARLY engagement
    # (first 4 weeks, i.e. date < 28), and breadth of engagement.
    vle = pd.read_csv(
        f"{UP}/studentVle.csv",
        usecols=["code_module", "code_presentation", "id_student",
                 "date", "sum_click"],
        dtype={"date": np.int32, "sum_click": np.int32},
    )
    keys = ["code_module", "code_presentation", "id_student"]
    g = vle.groupby(keys)
    agg = g.agg(
        total_clicks=("sum_click", "sum"),
        days_active=("date", "nunique"),
        first_day=("date", "min"),
    ).reset_index()
    early = (vle[vle["date"] < 28].groupby(keys)["sum_click"]
             .sum().rename("clicks_first4wk").reset_index())
    agg = agg.merge(early, on=keys, how="left")
    agg["clicks_first4wk"] = agg["clicks_first4wk"].fillna(0)

    df = info.merge(agg, on=keys, how="left")
    # Students with no VLE record at all = zero engagement
    for c in ["total_clicks", "days_active", "clicks_first4wk"]:
        df[c] = df[c].fillna(0)
    df["first_day"] = df["first_day"].fillna(999)  # never accessed

    # Binary target: at-risk = Fail or Withdrawn
    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    num = ["num_of_prev_attempts", "studied_credits", "total_clicks",
           "days_active", "clicks_first4wk", "first_day"]
    cat = ["gender", "region", "highest_education", "imd_band",
           "age_band", "disability", "code_module"]
    df["imd_band"] = df["imd_band"].fillna("Missing")
    X = df[num + cat]
    y = df["at_risk"]
    return X, y, num, cat


# ----------------------------------------------------------------------
# 2. Student Performance (Cortez): Maths + Portuguese combined
# ----------------------------------------------------------------------
def build_cortez():
    mat = pd.read_csv(f"{UP}/student-mat.csv", sep=";")
    por = pd.read_csv(f"{UP}/student-por.csv", sep=";")
    mat["subject"], por["subject"] = "maths", "portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    df["G3"] = pd.to_numeric(df["G3"])

    # At-risk = final grade below the Portuguese pass mark of 10.
    # G1 and G2 are EXCLUDED to avoid label leakage.
    df["at_risk"] = (df["G3"] < 10).astype(int)

    num = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health",
           "absences"]
    cat = ["school", "sex", "address", "famsize", "Pstatus", "Mjob",
           "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid",
           "activities", "nursery", "higher", "internet", "romantic",
           "subject"]
    X = df[num + cat]
    y = df["at_risk"]
    return X, y, num, cat


# ----------------------------------------------------------------------
# 3. Modelling
# ----------------------------------------------------------------------
def run_models(X, y, num, cat, name):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RNG, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
            random_state=RNG),
    }
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG)

    results, fitted = [], {}
    for mname, model in models.items():
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        results.append({
            "dataset": name, "model": mname,
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba),
            "confusion": confusion_matrix(yte, pred).tolist(),
        })
        fitted[mname] = pipe
    return results, fitted, (Xtr, Xte, ytr, yte)


def rf_importances(pipe, num, cat, top=10):
    """Aggregate one-hot importances back to original feature names."""
    pre = pipe.named_steps["pre"]
    rf = pipe.named_steps["clf"]
    names = list(num)
    ohe = pre.named_transformers_["cat"]
    for c, cats in zip(cat, ohe.categories_):
        names += [f"{c}" for _ in cats]
    imp = pd.Series(rf.feature_importances_, index=names)
    return imp.groupby(level=0).sum().sort_values(ascending=False).head(top)


if __name__ == "__main__":
    all_results = []

    print("Building OULAD features (aggregating 10.65M VLE rows)...")
    Xo, yo, num_o, cat_o = build_oulad()
    print(f"OULAD: {Xo.shape[0]} students, at-risk rate {yo.mean():.3f}")
    res_o, fit_o, _ = run_models(Xo, yo, num_o, cat_o, "OULAD")
    all_results += res_o

    print("Building Student Performance features...")
    Xc, yc, num_c, cat_c = build_cortez()
    print(f"Cortez: {Xc.shape[0]} students, at-risk rate {yc.mean():.3f}")
    res_c, fit_c, _ = run_models(Xc, yc, num_c, cat_c, "Student Performance")
    all_results += res_c

    print("\n=== RESULTS ===")
    for r in all_results:
        print(f"\n{r['dataset']} - {r['model']}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"  {k:10s}: {r[k]:.3f}")
        print(f"  confusion : {r['confusion']}")

    print("\n=== RF FEATURE IMPORTANCES (top 10, one-hot aggregated) ===")
    imp_o = rf_importances(fit_o["Random Forest"], num_o, cat_o)
    imp_c = rf_importances(fit_c["Random Forest"], num_c, cat_c)
    print("\nOULAD:\n", imp_o.round(3))
    print("\nStudent Performance:\n", imp_c.round(3))

    # Save for figure/table generation
    pd.DataFrame(all_results).to_json("/home/claude/task1/results.json")
    imp_o.to_json("/home/claude/task1/imp_oulad.json")
    imp_c.to_json("/home/claude/task1/imp_cortez.json")
    print("\nSaved results.")
