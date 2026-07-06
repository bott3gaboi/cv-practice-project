import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# ищем все results.csv внутри папки results
csv_files = list(RESULTS_DIR.rglob("results.csv"))

summary = []

for path in csv_files:
    run_name = path.parent.name

    # берем только emergency-модели
    if "emergency" not in run_name:
        continue

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    last = df.iloc[-1]

    precision = last.get("metrics/precision(B)")
    recall = last.get("metrics/recall(B)")
    map50 = last.get("metrics/mAP50(B)")
    map5095 = last.get("metrics/mAP50-95(B)")

    if precision is None or recall is None or map50 is None:
        continue

    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    summary.append({
        "Модель": run_name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50-95": round(map5095, 4),
        "Путь": str(path)
    })

summary_df = pd.DataFrame(summary)

# чтобы модели шли красиво
summary_df = summary_df.sort_values("Модель")

print(summary_df)

out_dir = BASE_DIR / "results" / "tables"
out_dir.mkdir(parents=True, exist_ok=True)

summary_df.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

# Excel сохраняем только если openpyxl установлен
try:
    summary_df.to_excel(out_dir / "model_comparison.xlsx", index=False)
except Exception as e:
    print("Excel не сохранен:", e)

plots_dir = BASE_DIR / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 5))
plt.bar(summary_df["Модель"], summary_df["mAP50"])
plt.ylabel("mAP50")
plt.title("Сравнение моделей по mAP50")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(plots_dir / "map50_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 5))
plt.bar(summary_df["Модель"], summary_df["Precision"])
plt.ylabel("Precision")
plt.title("Сравнение моделей по Precision")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(plots_dir / "precision_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 5))
plt.bar(summary_df["Модель"], summary_df["Recall"])
plt.ylabel("Recall")
plt.title("Сравнение моделей по Recall")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(plots_dir / "recall_comparison.png", dpi=300)
plt.show()