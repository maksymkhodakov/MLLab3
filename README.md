# Лабораторна робота №3 
# з дисципліни "Машинне навчання"
# студента групи ШІ Ходакова Максима Олеговича

### Datasets:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

# Висновки за результатами досліджень (SVM + базові моделі)

## 1) Breast Cancer Wisconsin (569×30, бінарна класифікація: malignant/benign)

**Коротко про дані**  
- 30 числових ознак, багато корельованих «сімейств» (mean / error / worst для radius, area, perimeter, smoothness тощо).  
- Баланс класів помірний (212 vs 357).  
- Візуальний аналіз (PCA, KDE) показує майже лінійну віддільність.  

**Підсумок по моделях і ядрах**

| Модель / Ядро | Найкращі гіперпараметри | Якість на TEST |
|---------------|--------------------------|----------------|
| Logistic Regression | C = 1.0 | **≈ 0.99** accuracy, macro F1 ≈ 0.99 |
| SVM (linear) | C = 0.01 | ≈ 0.98 accuracy |
| SVM (poly) | C = 100, degree = 3 | ≈ 0.96 accuracy |
| SVM (rbf) | C = 10, gamma = 0.01 | ≈ 0.97 accuracy |
| SVM (sigmoid) | C = 10, gamma = 0.03 | ≈ 0.92 accuracy |
| PolyFeatures(2)+SVM (rbf) | C = 100, gamma = scale | ≈ 0.94 accuracy |

**Висновок**  
- Дані **майже лінійно відокремлювані** → лінійні моделі дають найкращий результат.  
- **RBF/Poly** не дають виграшу, чутливі до `C, gamma`.  
- **PolyFeatures** погіршують якість (оверфіт).  
- Кращі моделі: Logistic Regression та Linear SVM.  

---

## 2) Digits (8×8, 1797×64, мультиклас 0–9)

**Коротко про дані**  
- 64 ознаки — інтенсивності пікселів 8×8.  
- 10 класів, дані добре збалансовані.  
- PCA-2D показує нелінійну структуру.  

**Підсумок по моделях і ядрах**

| Модель / Ядро | Найкращі гіперпараметри | Якість на TEST |
|---------------|--------------------------|----------------|
| Logistic Regression (multinomial) | C = 1000 | **≈ 0.95** accuracy |
| SVM (linear) | C = 0.01 | ≈ 0.97 accuracy |
| SVM (poly) | C = 100, degree = 3 | **≈ 0.99** accuracy |
| SVM (rbf) | C = 1, gamma = 0.03 | ≈ 0.98 accuracy |
| SVM (sigmoid) | C = 1, gamma = 0.01 | ≈ 0.97 accuracy |
| RBFSampler+Linear SVM | γ=0.05, n=800, C=1 | ≈ 0.95 accuracy |

**Висновок**  
- Нелінійна природа даних → найкраще працюють **SVM poly** та **SVM rbf**.  
- **RBFSampler** дає компроміс між швидкістю і якістю (≈0.95).  
- Logistic Regression слабша, але проста і швидка.  

---

## Загальні висновки

1. **Геометрія даних визначає ядро**:  
   - Breast Cancer — майже лінійний → LogReg, Linear SVM.  
   - Digits — нелінійний → Poly/RBF.  

2. **Регуляризація критична**: великі `C` та `γ` призводять до оверфіту.  

3. **Розширення ознак**:  
   - Допомагає для складних даних (Digits, RFF).  
   - Шкодить для простих (Breast Cancer, PolyFeatures).  

4. **Метрики і графіки**:  
   - Macro F1 + Confusion Matrix для мультикласу.  
   - Learning Curves → контроль overfit/underfit.  

5. **Практичні поради**:  
   - Breast Cancer: достатньо Logistic Regression / Linear SVM.  
   - Digits: для якості — SVM (poly/rbf), для швидкості — Logistic Regression / RFF+Linear.
