"""
SVM на Breast Cancer Wisconsin (569 зразків, 30 фіч)
Сценарій: "мало точок — багато фіч"

Структура:
1) EDA: розмір, баланс класів, heatmap кореляцій (топ фіч).
2) Базовий лінійний аналіз: Logistic Regression (базлайн), (Діагностично) Linear Regression.
3) Валід. тюнінг гіперпараметрів для SVM (linear, poly, rbf, sigmoid) + для лог. регресії.
   - Тренуємо на train, перевіряємо F1(macro) на val, обираємо кращі гіперпараметри.
   - Потім донавчаємо на (train+val) з обраними гіперпараметрами й оцінюємо на test.
4) Розширення ознак: PolynomialFeatures(deg=2) + SVM RBF (аналог «додати нелінійність у фічі»).
5) Візуалізації:
   - Confusion Matrix (test)
   - Learning Curve (тільки train+val, без test, щоб не «підглядати»)
   - Validation Curve по C (малюємо на валідації)
6) Перевірка overfit/underfit: різниця F1(train) vs F1(val) під час тюнінгу + поведінка кривих.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, precision_score, recall_score
)

# ===============================
# 0) Налаштування/константи
# ===============================
RANDOM_SEED = 42
SPLIT_TRAIN = 0.60  # глобальний train
SPLIT_VAL = 0.20  # глобальна валідація (для тюнінгу гіперпараметрів)
SPLIT_TEST = 0.20  # глобальний test (тільки фінальна оцінка)
assert abs(SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST - 1.0) < 1e-9

# Для Learning Curve: робимо стратифікований ShuffleSplit тільки по (train+val)
LCV = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
TRAIN_FRACTIONS = np.linspace(0.3, 1.0, 8)  # від 30% до 100% від (train+val)

# ===============================
# 1) Дані та EDA
# ===============================
print("ЕТАП 1: Завантаження Breast Cancer")
ds = load_breast_cancer(as_frame=True)
X_all = ds.data  # 30 числових фіч
y_all = ds.target  # 0 = malignant, 1 = benign
class_names = ds.target_names  # ['malignant' 'benign']
print("Shape:", X_all.shape, "| classes:", dict(zip(class_names, np.bincount(y_all))))

print("Опис датасету:")
print(" • Breast Cancer Wisconsin — медичний датасет: 569 зразків, 30 числових фіч, бінарні класи: malignant (0), benign (1).")
print(" • Кожна фіча — статистика текстури клітин на зображеннях біопсії (mean / se / worst для radius, texture, perimeter, area, smoothness тощо).")
print("\nСписок фіч:")
all_feats = list(X_all.columns)
print(" ", ", ".join(all_feats), "...")
print(f"Всього фіч: {len(all_feats)}")

# Heatmap кореляцій (обмежимося топ-25 змінними за дисперсією, щоб графік був читабельний)
top_vars = X_all.var().sort_values(ascending=False).head(25).index
plt.figure(figsize=(10, 8))
sns.heatmap(X_all[top_vars].corr(), cmap="coolwarm", center=0, square=False)
plt.title("Кореляційна матриця (топ-25 фіч за дисперсією)")
plt.tight_layout();
plt.show()

# --- «Приклади» для не-зображень: PCA 2D + розподіли кількох фіч ---

# PCA 2D (щоб «побачити» структуру даних, аналог зображень для Digits)
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
X2 = pca2.fit_transform(X_all)
plt.figure(figsize=(6,5))
sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=y_all, palette=["tab:red", "tab:green"], s=25, alpha=0.8, legend=True)
plt.title("PCA 2D проєкція (Breast Cancer): malignant vs benign")
plt.tight_layout(); plt.show()

# «Міні-портрети ознак» — розподіли кількох ключових фіч (по класах)
key_feats = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
available_feats = [f for f in key_feats if f in X_all.columns]
n = len(available_feats)
if n > 0:
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4*cols, 3*rows))
    for i, f in enumerate(available_feats, 1):
        plt.subplot(rows, cols, i)
        sns.kdeplot(x=X_all[f], hue=y_all, fill=True, common_norm=False, alpha=0.4,
                    palette=["tab:red", "tab:green"], linewidth=1)
        plt.title(f)
        plt.xlabel(f); plt.ylabel("Щільність")
    plt.suptitle("Розподіли ознак за класами (malignant=черв., benign=зел.)")
    plt.tight_layout(); plt.show()

# Показати кілька «пацієнтів» (перші 5 рядків)
print("\nПерші 5 рядків (фрагмент даних):")
print(X_all.head(5).assign(label=y_all[:5]).to_string(index=False))


# ===============================
# 2) Розбиття на Train / Val / Test
# ===============================
print("\nЕТАП 2: Розбиття 60/20/20 (стратифіковано)")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=(1 - SPLIT_TRAIN),
    random_state=RANDOM_SEED, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=SPLIT_TEST / (SPLIT_VAL + SPLIT_TEST),
    random_state=RANDOM_SEED, stratify=y_temp
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Допоміжні множини (для LearningCurve ми будемо використовувати train+val)
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)


# ===============================
# 3) Утиліти (метрики, графіки, тюнінг)
# ===============================
def summarize_split_metrics(y_true, y_pred, split_name):
    """Друкуємо короткий summary по основних метриках для зручності."""
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    pm = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rm = recall_score(y_true, y_pred, average="macro", zero_division=0)
    print(
        f"[{split_name}] Acc={acc:.3f} | F1(macro)={f1m:.3f} | F1(weighted)={f1w:.3f} | Prec(m)={pm:.3f} | Rec(m)={rm:.3f}")


def confusion_on_test(name, estimator):
    """Малюємо матрицю плутанини на тесті."""
    y_pred_test = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (TEST) — {name}")
    plt.tight_layout();
    plt.show()


def plot_learning_curve(title, pipeline):
    """Будуємо Learning Curve на (train+val) — test не використовуємо."""
    print(f"Побудова Learning Curve: {title}")
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_trainval, y_trainval,
        train_sizes=TRAIN_FRACTIONS,
        cv=LCV, scoring="f1_macro", error_score=np.nan
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.nanmean(train_scores, axis=1), "o-", label="Train F1")
    plt.plot(train_sizes, np.nanmean(val_scores, axis=1), "o-", label="Val F1 (inner)")
    plt.xlabel("К-сть тренувальних зразків (із train+val)")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.grid(alpha=0.3);
    plt.legend();
    plt.tight_layout();
    plt.show()


def manual_val_search(name, pipe_maker, param_grid):
    """
    Простіший за GridSearchCV «ручний» тюнінг по валідації:
    - для кожної комбінації гіперпараметрів:
        * навчаємо на TRAIN
        * вимірюємо F1(macro) на VAL
    - обираємо комбінацію з найвищим F1 на VAL
    - донавчаємо на (TRAIN+VAL) з обраними параметрами
    - повертаємо донавчений естиматор, кращі параметри, та логи.
    """
    print(f"\nЕТАП тюнінгу (validation search): {name}")
    best_score = -np.inf
    best_params = None
    best_estimator = None

    # перебираємо просту сітку параметрів
    for params in param_grid:
        pipe = pipe_maker(**params)
        pipe.fit(X_train, y_train)
        # метрики на Train/Val
        y_pred_tr = pipe.predict(X_train)
        y_pred_va = pipe.predict(X_val)
        f1_tr = f1_score(y_train, y_pred_tr, average="macro", zero_division=0)
        f1_va = f1_score(y_val, y_pred_va, average="macro", zero_division=0)
        gap = f1_tr - f1_va

        print(f"  params={params} | F1(TR)={f1_tr:.3f}  F1(VAL)={f1_va:.3f}  GAP={gap:+.3f}")

        if f1_va > best_score:
            best_score = f1_va
            best_params = params
            best_estimator = pipe

    print(f"Кращі параметри (за VAL F1): {best_params} | VAL F1={best_score:.3f}")

    # донавчаємо на (train+val) з кращими гіперпараметрами
    final_estimator = pipe_maker(**best_params)
    final_estimator.fit(X_trainval, y_trainval)

    return final_estimator, best_params


# ===============================
# 4) Базовий лінійний аналіз
# ===============================
print("\nЕТАП 3: Базові моделі (перед SVM)")


# 4.1 Logistic Regression — наш лінійний базлайн
def make_logreg(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=C, random_state=RANDOM_SEED))
    ])


logreg_grid = [{"C": c} for c in np.logspace(-3, 3, 7)]
logreg_best, logreg_best_params = manual_val_search("Logistic Regression", make_logreg, logreg_grid)

# Оцінка на TEST
y_pred_test = logreg_best.predict(X_test)
print("\n[LogisticRegression] Звіт на TEST:")
print(classification_report(y_test, y_pred_test, target_names=class_names))
confusion_on_test(f"LogReg C={logreg_best_params['C']}", logreg_best)
plot_learning_curve(f"Learning Curve — Logistic Regression (C={logreg_best_params['C']})", logreg_best)

# 4.2 (Діагностика) Лінійна регресія з округленням — НЕ класифікатор, лише демонстрація
linreg = Pipeline([("scaler", StandardScaler()), ("clf", LinearRegression())])
linreg.fit(X_train, y_train)
y_lr_pred_test = np.clip(np.rint(linreg.predict(X_test)), 0, 1).astype(int)
print("\n[Діагностика] LinearRegression→round: TEST Accuracy =", accuracy_score(y_test, y_lr_pred_test))

# ===============================
# 5) SVM: різні ядра + тюнінг на валідації
# ===============================
print("\nЕТАП 4: SVM (ядра linear / poly / rbf / sigmoid)")


# 5.1 SVM Linear (LinearSVC) — швидкий лінійний SVM
def make_svm_linear(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=C, random_state=RANDOM_SEED))
    ])


svm_lin_grid = [{"C": c} for c in np.logspace(-3, 3, 7)]
svm_lin_best, svm_lin_params = manual_val_search("SVM Linear", make_svm_linear, svm_lin_grid)
print("\n[SVM Linear] TEST:")
print(classification_report(y_test, svm_lin_best.predict(X_test), target_names=class_names))
confusion_on_test(f"SVM Linear C={svm_lin_params['C']}", svm_lin_best)
plot_learning_curve(f"Learning Curve — SVM Linear (C={svm_lin_params['C']})", svm_lin_best)


# 5.2 SVM Poly (ступінь полінома + C)
def make_svm_poly(C=1.0, degree=3, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", degree=degree, C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_poly_grid = [{"C": c, "degree": d, "gamma": "scale"} for c in np.logspace(-2, 2, 5) for d in [2, 3, 4]]
svm_poly_best, svm_poly_params = manual_val_search("SVM Poly", make_svm_poly, svm_poly_grid)
print("\n[SVM Poly] TEST:")
print(classification_report(y_test, svm_poly_best.predict(X_test), target_names=class_names))
confusion_on_test(f"SVM Poly C={svm_poly_params['C']} deg={svm_poly_params['degree']}", svm_poly_best)
plot_learning_curve(f"Learning Curve — SVM Poly (C={svm_poly_params['C']}, deg={svm_poly_params['degree']})",
                    svm_poly_best)


# 5.3 SVM RBF (C, gamma) — зазвичай найсильніший
def make_svm_rbf(C=1.0, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_rbf_grid = [{"C": c, "gamma": g} for c in np.logspace(-2, 2, 5) for g in [0.01, 0.03, 0.1, 0.3, "scale"]]
svm_rbf_best, svm_rbf_params = manual_val_search("SVM RBF", make_svm_rbf, svm_rbf_grid)
print("\n[SVM RBF] TEST:")
print(classification_report(y_test, svm_rbf_best.predict(X_test), target_names=class_names))
confusion_on_test(f"SVM RBF C={svm_rbf_params['C']} gamma={svm_rbf_params['gamma']}", svm_rbf_best)
plot_learning_curve(f"Learning Curve — SVM RBF (C={svm_rbf_params['C']}, gamma={svm_rbf_params['gamma']})",
                    svm_rbf_best)


# 5.4 SVM Sigmoid (C, gamma) — рідко кращий, але для повноти
def make_svm_sigmoid(C=1.0, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="sigmoid", C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_sig_grid = [{"C": c, "gamma": g} for c in np.logspace(-2, 2, 5) for g in [0.01, 0.03, 0.1, 0.3, "scale"]]
svm_sig_best, svm_sig_params = manual_val_search("SVM Sigmoid", make_svm_sigmoid, svm_sig_grid)
print("\n[SVM Sigmoid] TEST:")
print(classification_report(y_test, svm_sig_best.predict(X_test), target_names=class_names))
confusion_on_test(f"SVM Sigmoid C={svm_sig_params['C']} gamma={svm_sig_params['gamma']}", svm_sig_best)
plot_learning_curve(f"Learning Curve — SVM Sigmoid (C={svm_sig_params['C']}, gamma={svm_sig_params['gamma']})",
                    svm_sig_best)

# ===============================
# 6) Розширення ознак: PolyFeatures + SVM RBF
# ===============================
print("\nЕТАП 5: Розширення ознак (PolynomialFeatures degree=2) + SVM RBF")


def make_svm_rbf_polyfeats(C=1.0, gamma="scale", degree=2):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False — безпечно, коли можуть бути розріджені матриці
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_rbf_poly_grid = [{"C": c, "gamma": g, "degree": 2} for c in np.logspace(-2, 2, 5) for g in
                     [0.01, 0.03, 0.1, 0.3, "scale"]]
svm_rbf_poly_best, svm_rbf_poly_params = manual_val_search("SVM RBF + PolyFeatures(2)", make_svm_rbf_polyfeats,
                                                           svm_rbf_poly_grid)
print("\n[SVM RBF + Poly(2)] TEST:")
print(classification_report(y_test, svm_rbf_poly_best.predict(X_test), target_names=class_names))
confusion_on_test(f"SVM RBF + Poly(2) C={svm_rbf_poly_params['C']} gamma={svm_rbf_poly_params['gamma']}",
                  svm_rbf_poly_best)
plot_learning_curve(
    f"Learning Curve — SVM RBF + Poly(2) (C={svm_rbf_poly_params['C']}, gamma={svm_rbf_poly_params['gamma']})",
    svm_rbf_poly_best)

print("\nГотово ✅")
