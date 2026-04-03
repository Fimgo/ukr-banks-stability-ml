import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Зчитуємо CSV-файл з даними про банки у таблицю DataFrame.
df = pd.read_csv("banks_data.csv")

# Перевіряємо, чи в наборі даних є всі потрібні колонки.
required_columns = ["Bank", "Total_Assets", "Equity", "Net_Profit"]
missing_columns = [column for column in required_columns if column not in df.columns]

# Якщо якоїсь колонки немає, зупиняємо програму і показуємо помилку.
if missing_columns:
    raise ValueError(
        f"У файлі banks_data.csv відсутні колонки: {', '.join(missing_columns)}"
    )

# Залишаємо лише потрібні колонки для аналізу.
df = df[required_columns].copy()

# Нормалізуємо назви банків, щоб прибрати дублікати з різним регістром або зайвими пробілами.
df["Bank_Normalized"] = df["Bank"].astype(str).str.strip().str.upper()

# Видаляємо дублікати банків за нормалізованою назвою.
df = df.drop_duplicates(subset=["Bank_Normalized"]).copy()

# Прибираємо технічну колонку після очищення даних.
df = df.drop(columns=["Bank_Normalized"])

# Перетворюємо числові колонки у числовий формат.
numeric_columns = ["Total_Assets", "Equity", "Net_Profit"]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Видаляємо рядки з пропущеними значеннями у ключових числових колонках.
df = df.dropna(subset=numeric_columns)

# Видаляємо рядки, де активи дорівнюють нулю, щоб уникнути ділення на нуль.
df = df[df["Total_Assets"] != 0].copy()

# Перевіряємо, чи залишилося достатньо банків для поділу на 3 кластери.
if len(df) < 3:
    raise ValueError("Після очищення даних залишилося менше 3 банків. Кластеризацію виконати неможливо.")

# Обчислюємо ROA: прибуток / активи.
df["ROA"] = df["Net_Profit"] / df["Total_Assets"]

# Обчислюємо адекватність капіталу: капітал / активи.
df["Capital_Adequacy"] = df["Equity"] / df["Total_Assets"]

# Формуємо таблицю ознак для кластеризації.
features = df[["ROA", "Capital_Adequacy"]]

# Масштабуємо ознаки для коректної роботи K-Means.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Реалізуємо метод ліктя: порівнюємо інерцію моделі для різної кількості кластерів.
max_clusters = min(8, len(features))
elbow_data = []

for clusters_count in range(1, max_clusters + 1):
    elbow_model = KMeans(n_clusters=clusters_count, random_state=42, n_init=10)
    elbow_model.fit(scaled_features)
    elbow_data.append(
        {"Clusters": clusters_count, "Inertia": elbow_model.inertia_}
    )

# Зберігаємо дані методу ліктя в окрему таблицю.
elbow_table = pd.DataFrame(elbow_data)
elbow_table.to_csv("elbow_method.csv", index=False, encoding="utf-8-sig")

# Створюємо модель K-Means для поділу банків на 3 кластери.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Визначаємо номер кластера для кожного банку.
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Створюємо окрему підсумкову таблицю з результатами аналізу.
result_table = df[
    ["Bank", "Total_Assets", "Equity", "Net_Profit", "ROA", "Capital_Adequacy", "Cluster"]
].copy()

# Сортуємо банки за номером кластера для більш зручного перегляду.
result_table = result_table.sort_values(["Cluster", "Bank"]).reset_index(drop=True)

# Розраховуємо середні значення показників у кожному кластері.
cluster_means = (
    result_table.groupby("Cluster")[["ROA", "Capital_Adequacy"]]
    .mean()
    .reset_index()
)

# Зберігаємо підсумкову таблицю та середні значення у файли.
result_table.to_csv("banks_clustered.csv", index=False, encoding="utf-8-sig")
cluster_means.to_csv("cluster_means.csv", index=False, encoding="utf-8-sig")

# Виводимо в консоль коротку підсумкову інформацію.
print("Список банків і номери їхніх кластерів:")
print(result_table[["Bank", "Cluster"]].to_string(index=False))

print("\nРозраховані показники:")
print(result_table[["Bank", "ROA", "Capital_Adequacy", "Cluster"]].to_string(index=False))

print("\nСередні значення показників по кластерах:")
print(cluster_means.to_string(index=False))

print("\nПідсумкову таблицю збережено у файл banks_clustered.csv")
print("Середні значення по кластерах збережено у файл cluster_means.csv")
print("Дані для методу ліктя збережено у файл elbow_method.csv")

# Налаштовуємо стиль графіків.
sns.set(style="whitegrid")

# Створюємо фігуру з двома графіками: метод ліктя та кластеризація банків.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Будуємо графік методу ліктя.
axes[0].plot(
    elbow_table["Clusters"],
    elbow_table["Inertia"],
    marker="o",
    linewidth=2,
)
axes[0].set_title("Метод ліктя для вибору кількості кластерів")
axes[0].set_xlabel("Кількість кластерів")
axes[0].set_ylabel("Інерція")
axes[0].set_xticks(elbow_table["Clusters"])

# Будуємо точковий графік, де колір відповідає кластеру.
sns.scatterplot(
    data=result_table,
    x="ROA",
    y="Capital_Adequacy",
    hue="Cluster",
    palette="Set1",
    s=120,
    ax=axes[1],
)

# Додаємо назву графіка і підписи осей.
axes[1].set_title("Кластеризація банків за фінансовою стійкістю")
axes[1].set_xlabel("ROA (прибуток / активи)")
axes[1].set_ylabel("Адекватність капіталу (капітал / активи)")
axes[1].legend(title="Кластер")

# Робимо автоматичне вирівнювання елементів фігури.
plt.tight_layout()

# Зберігаємо спільну фігуру у файл.
plt.savefig("cluster_analysis.png", dpi=300, bbox_inches="tight")

# Показуємо графіки на екрані.
plt.show()
