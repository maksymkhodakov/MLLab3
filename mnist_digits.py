"""
SVM на Digits (8x8, 1797 зразків, 64 фічі)
Сценарій: "мало фіч — багато точок" (але 10 класів)

1) EDA: приклади зображень, PCA-2D візуалізація.
2) Базовий лінійний аналіз: Logistic Regression (multinomial baseline).
3) Валід. тюнінг гіперпараметрів: SVM linear / poly / rbf / sigmoid.
4) Розширення ознак: Random Fourier Features (RBFSampler) + Linear SVM
   (це явне наближення RBF-ядра у просторі фіч).
5) Візуалізації та overfit/underfit: Confusion Matrix (test), Learning Curve (train+val).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, precision_score, recall_score
)

# ===============================
# 0) Налаштування
# ===============================
RANDOM_SEED = 42
SPLIT_TRAIN = 0.60
SPLIT_VAL = 0.20
SPLIT_TEST = 0.20
assert abs(SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST - 1.0) < 1e-9

LCV = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_SEED)
TRAIN_FRACTIONS = np.linspace(0.1, 1.0, 10)  # датасет великий — можна починати з 10%

# ===============================
# 1) Дані та EDA
# ===============================
print("ЕТАП 1: Завантаження Digits")
digits = load_digits()
X_all = digits.data  # (1797, 64)
y_all = digits.target  # класи 0..9
class_names = [str(c) for c in np.unique(y_all)]
print("Shape:", X_all.shape, "| класів:", len(class_names))

print("Опис датасету:")
print(" • Digits 8×8 — рукописні цифри (0–9), 1797 зразків, 64 пікселі-ознаки (інтенсивність 0..16).")
print(" • Кожна фіча — яскравість пікселя у розгортці 8×8 (рядковий порядок).")

feature_names_digits = [f"pixel_{i}" for i in range(X_all.shape[1])]
print("\nСписок фіч (перші 20):")
print(" ", ", ".join(feature_names_digits[:20]), "...")
print(f"Всього фіч: {len(feature_names_digits)}")

# Показати кілька зображень
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for ax, img, label in zip(axes.ravel(), digits.images[:16], y_all[:16]):
    ax.imshow(img, cmap="gray");
    ax.set_title(label);
    ax.axis("off")
plt.suptitle("Перші 16 зображень Digits");
plt.tight_layout();
plt.show()

# PCA 2D — просто для інтуїції щодо структури
pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
X2 = pca2.fit_transform(X_all)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=y_all, palette="tab10", s=18, legend=False)
plt.title("PCA 2D проєкція (Digits)");
plt.tight_layout();
plt.show()

# ===============================
# 2) Розбиття Train / Val / Test
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

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])


# ===============================
# 3) Утиліти: метрики, графіки, тюнінг
# ===============================
def summarize_split_metrics(y_true, y_pred, split_name):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    pm = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rm = recall_score(y_true, y_pred, average="macro", zero_division=0)
    print(
        f"[{split_name}] Acc={acc:.3f} | F1(macro)={f1m:.3f} | F1(weighted)={f1w:.3f} | Prec(m)={pm:.3f} | Rec(m)={rm:.3f}")


def confusion_on_test(name, estimator):
    y_pred_test = estimator.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (TEST) — {name}")
    plt.tight_layout();
    plt.show()


def plot_learning_curve(title, pipeline):
    print(f"Побудова Learning Curve: {title}")
    tr_sizes, tr_scores, va_scores = learning_curve(
        pipeline, X_trainval, y_trainval,
        train_sizes=TRAIN_FRACTIONS,
        cv=LCV, scoring="f1_macro", error_score=np.nan
    )
    plt.figure(figsize=(8, 5))
    plt.plot(tr_sizes, np.nanmean(tr_scores, axis=1), "o-", label="Train F1")
    plt.plot(tr_sizes, np.nanmean(va_scores, axis=1), "o-", label="Val F1 (inner)")
    plt.xlabel("К-сть тренувальних зразків (із train+val)")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.grid(alpha=0.3);
    plt.legend();
    plt.tight_layout();
    plt.show()


def manual_val_search(name, pipe_maker, param_grid):
    print(f"\nЕТАП тюнінгу (validation search): {name}")
    best_score = -np.inf
    best_params = None

    for params in param_grid:
        pipe = pipe_maker(**params)
        pipe.fit(X_train, y_train)
        y_pred_tr = pipe.predict(X_train)
        y_pred_va = pipe.predict(X_val)
        f1_tr = f1_score(y_train, y_pred_tr, average="macro", zero_division=0)
        f1_va = f1_score(y_val, y_pred_va, average="macro", zero_division=0)
        gap = f1_tr - f1_va
        print(f"  params={params} | F1(TR)={f1_tr:.3f}  F1(VAL)={f1_va:.3f}  GAP={gap:+.3f}")

        if f1_va > best_score:
            best_score = f1_va
            best_params = params

    print(f"Кращі параметри (за VAL F1): {best_params} | VAL F1={best_score:.3f}")

    # донавчання на train+val з найкращими параметрами
    final_est = pipe_maker(**best_params)
    final_est.fit(X_trainval, y_trainval)
    return final_est, best_params


# ===============================
# 4) Базова Logistic Regression (multinomial)
# ===============================
print("\nЕТАП 3: Logistic Regression (multinomial) — базлайн")


def make_logreg(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=RANDOM_SEED))
    ])


logreg_grid = [{"C": c} for c in np.logspace(-3, 3, 7)]
logreg_best, logreg_best_params = manual_val_search("Logistic Regression", make_logreg, logreg_grid)
print("\n[LogReg] TEST:")
print(classification_report(y_test, logreg_best.predict(X_test)))
confusion_on_test(f"LogReg C={logreg_best_params['C']}", logreg_best)
plot_learning_curve(f"Learning Curve — Logistic Regression (C={logreg_best_params['C']})", logreg_best)

# ===============================
# 5) SVM: різні ядра + тюнінг на валідації
# ===============================
print("\nЕТАП 4: SVM (ядра linear / poly / rbf / sigmoid)")


# 5.1 Linear SVM
def make_svm_linear(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=C, random_state=RANDOM_SEED))
    ])


svm_lin_grid = [{"C": c} for c in np.logspace(-3, 3, 7)]
svm_lin_best, svm_lin_params = manual_val_search("SVM Linear", make_svm_linear, svm_lin_grid)
print("\n[SVM Linear] TEST:")
print(classification_report(y_test, svm_lin_best.predict(X_test)))
confusion_on_test(f"SVM Linear C={svm_lin_params['C']}", svm_lin_best)
plot_learning_curve(f"Learning Curve — SVM Linear (C={svm_lin_params['C']})", svm_lin_best)


# 5.2 Poly SVM
def make_svm_poly(C=1.0, degree=3, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="poly", degree=degree, C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_poly_grid = [{"C": c, "degree": d, "gamma": "scale"} for c in np.logspace(-2, 2, 5) for d in [2, 3, 4]]
svm_poly_best, svm_poly_params = manual_val_search("SVM Poly", make_svm_poly, svm_poly_grid)
print("\n[SVM Poly] TEST:")
print(classification_report(y_test, svm_poly_best.predict(X_test)))
confusion_on_test(f"SVM Poly C={svm_poly_params['C']} deg={svm_poly_params['degree']}", svm_poly_best)
plot_learning_curve(f"Learning Curve — SVM Poly (C={svm_poly_params['C']}, deg={svm_poly_params['degree']})",
                    svm_poly_best)


# 5.3 RBF SVM
def make_svm_rbf(C=1.0, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_rbf_grid = [{"C": c, "gamma": g} for c in np.logspace(-2, 2, 5) for g in [0.01, 0.03, 0.1, 0.3, "scale"]]
svm_rbf_best, svm_rbf_params = manual_val_search("SVM RBF", make_svm_rbf, svm_rbf_grid)
print("\n[SVM RBF] TEST:")
print(classification_report(y_test, svm_rbf_best.predict(X_test)))
confusion_on_test(f"SVM RBF C={svm_rbf_params['C']} gamma={svm_rbf_params['gamma']}", svm_rbf_best)
plot_learning_curve(f"Learning Curve — SVM RBF (C={svm_rbf_params['C']}, gamma={svm_rbf_params['gamma']})",
                    svm_rbf_best)


# 5.4 Sigmoid SVM
def make_svm_sigmoid(C=1.0, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="sigmoid", C=C, gamma=gamma, random_state=RANDOM_SEED))
    ])


svm_sig_grid = [{"C": c, "gamma": g} for c in np.logspace(-2, 2, 5) for g in [0.01, 0.03, 0.1, 0.3, "scale"]]
svm_sig_best, svm_sig_params = manual_val_search("SVM Sigmoid", make_svm_sigmoid, svm_sig_grid)
print("\n[SVM Sigmoid] TEST:")
print(classification_report(y_test, svm_sig_best.predict(X_test)))
confusion_on_test(f"SVM Sigmoid C={svm_sig_params['C']} gamma={svm_sig_params['gamma']}", svm_sig_best)
plot_learning_curve(f"Learning Curve — SVM Sigmoid (C={svm_sig_params['C']}, gamma={svm_sig_params['gamma']})",
                    svm_sig_best)

# ===============================
# 6) Розширення ознак: Random Fourier Features + Linear SVM
# ===============================
print("\nЕТАП 5: Random Fourier Features (RBFSampler) + Linear SVM")


def make_rff_linear_svm(gamma=0.1, n_components=500, C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rff", RBFSampler(gamma=gamma, n_components=n_components, random_state=RANDOM_SEED)),
        ("clf", LinearSVC(C=C, random_state=RANDOM_SEED))
    ])


rff_grid = [{"gamma": g, "n_components": n, "C": c}
            for g in [0.05, 0.1, 0.2]
            for n in [300, 500, 800]
            for c in [0.5, 1.0, 3.0]]
rff_best, rff_params = manual_val_search("RFF + LinearSVM", make_rff_linear_svm, rff_grid)
print("\n[RFF + LinearSVM] TEST:")
print(classification_report(y_test, rff_best.predict(X_test)))
confusion_on_test(f"RFF + LinearSVM (γ={rff_params['gamma']}, n={rff_params['n_components']}, C={rff_params['C']})",
                  rff_best)
plot_learning_curve(
    f"Learning Curve — RFF + LinearSVM (γ={rff_params['gamma']}, n={rff_params['n_components']}, C={rff_params['C']})",
    rff_best)

print("\nГотово ✅")
