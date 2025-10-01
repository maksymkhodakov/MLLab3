# Лабораторна робота №3 
# з дисципліни "Машинне навчання"
# студента групи ШІ Ходакова Максима Олеговича

### Datasets:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

```
Принцип роботи

Спліт 60/20/20: train → навчання, val → тюнінг, test → тільки фінальна перевірка.

Pipeline з StandardScaler: запобігає data leakage.

Grid Search вручну (manual_val_search): вибір найкращих параметрів за F1(macro).

SVM ядра:

Linear → перевірка, чи дані майже лінійні.

Poly → тестує поліноміальні межі.

RBF → універсальне ядро для складних даних.

Sigmoid → історично як нейромережа, рідко найкраще, але додає різноманіття.

Feature expansion: перевірка, чи розширення ознак дає кращу відокремлюваність.
```
