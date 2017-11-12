"""
Пример построение линейного многоклассового классификатора с использованием
функций потерь:
  Hinge Loss (SVM)
  Softmax (Logistic Regression)
"""

#Загрузка сторонних модулей
#модуль для матричных операций и математических функций
import numpy as np 

# модуль для построения графиков
import matplotlib.pyplot as plt

#загружаем субмодуль datasets из модуля sklearn, из него будем брать тестовые коллекции
from sklearn import datasets

#субмодуль, содержащий полезные функции для работы с выборками
from sklearn import model_selection

#субмодуль, содержащий функции для оценки качества
from sklearn import metrics


def label_to_one_hot(labels):
    """Перевод меток класса в one-hot вектор

    labels - массив меток

    Return:
    массив one-hot векторов, соответствующих меткам класса
    """

    # находим все уникальные метки
    valid_labels = set(labels)

    # создаем пусой бинарный массив
    one_hot = np.zeros((len(labels),len(valid_labels)), dtype=np.bool)

    # перебираем пары, составленные из меток классов (label) и их порядковым номером (i)
    for i, label in enumerate(valid_labels):
        # ставим 1 в соответствии с порядковым номером класса пользуясь возможностями бинарной
        # индексации в np.array
        one_hot[labels == label, i] = 1
    return one_hot


def hinge_loss(x, y, w):
    """Вычисление функции потерь для многоклассового SVM

    x - вектор признаков
    y - метка класса
    w - матрица весов

    Return:
    значение функции потерь
    """
    scores = w.dot(x.T)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = margins.sum()
    return loss_i

def softmax(x, y, w):
    """Вычисление функции потерь для логистической регрессии 

    x - вектор признаков
    y - метка класса
    w - матрица весов

    Return:
    значение функции потреь
    """
    scores = w.dot(x.T)
    exp_scores = np.exp(scores)
    prob = exp_scores[y] / exp_scores.sum()
    loss_i = -np.log(prob)
    return loss_i

def global_loss(X, Y, w, loss_fun):
    """Вычисление функции потерь по выборке

    X - матрица, строки которой соответствуют векторам признаков
    Y - вектор меток классов
    w - матрица весов
    loss_fun - функция потерь 

    Return:
    значение функции потерь по выборке
    """

    loss_sum = np.sum([loss_fun(x, y, w) for x, y in zip(X,Y)])
    return loss_sum

def sample_train_data(X, Y, batch_size):
    """Построение водвыборки заданного размера из выборки

    X - коллекция векторов-признаков
    Y - коллекция соответствующих меток
    batch_size - размер сэмплируемого батча. Если размер выборки меньше чем указанный
      размер батча, то возвращается выборка целиком

    Return:
    (sub_x, sub_y) - пара, составленная из построенных подвыборок векторов-признаков и меток классов
    """
    if len(Y) <= batch_size:
        return X,Y
    start = np.random.randint(len(Y) - batch_size)
    return X[start: start + batch_size], Y[start: start + batch_size]

def numeric_gradient(X, Y, w, loss_fun, delta=1.e-6):
    """Численная оценка градиента

    X - матрица, строки которой соответствуют векторам признаков
    Y - вектор меток классов
    w - матрица весов
    loss_fun - функция потерь
    delta - шаг для оценки частных производных

    Return:
    (grad, loss_val) - пара, составленная из градиента и значения функции потреь
    """
    
    # запоминаем значение функции потреь при текущей матрице весов w
    L0 = loss_fun(X,Y,w) 
    # представим матрицу весов в виде вектора
    w_vec = w.flatten() 
    k = len(w_vec)
    # ниже в цикле вычисляются частные производные
    # для этого мы строим единичную матрицу (на диагонали 1, все остальное 0) размера k х k: np.eye(k)
    # перебираем в цикле строки этой матрицы: 
    #   [ .. for q in np.eye(k)]
    # строим новый вектор весов, в котором дается приращение delta очередному коэффициенту вектора весов:
    #   (w_vec + delta*q)
    # приводим вектор весов обратно в матрицу, т.к. функция потерь принимает на вход матрицу:
    #   (w_vec + delta*q).reshape(w.shape)
    # вычисляем функцию потерь в точке с приращением:
    #   loss_fun(X, Y, (w_vec + delta*q).reshape(w.shape))
    # находим теперь приращение функции:
    #   loss_fun(X, Y, (w_vec + delta*q).reshape(w.shape)) - L0
    # найденные значения сохраняем в списке (конструкция [ .. for q in .. ]
    # конвертируем список в np.array, производим поэлементное деление приращение функции на приращение аргумента
    g = np.array([loss_fun(X, Y, (w_vec + delta*q).reshape(w.shape)) - L0 for q in np.eye(k)]) / delta
    
    # приводим градиент к той же форме что и матрица весов (для дальнейшего удобства)
    g = g.reshape(w.shape) 

    # возвращаем пару: градиент, значение функции потреь
    return g, L0


def sgd(X, Y, w, grad_fun, batch_size=128, n_iter=1000, step=1.e-7):
    """Стахостический градиентный спуск

    X - матрица, строки которой соответствуют векторам признаков
    Y - вектор меток классов
    w - матрица весов
    grad_fun - функция фычисления градиента
    batch_size - размер батча, используемого для оценки градиента
    n_iter - количество итераций
    step - шаг по градиенту

    Return:
    best_w - матрица весов, соответствующая минимальному значению функции потерь
    """
    best_loss = np.inf
    for i in range(n_iter):
        # строим подвыборку для оценки градиента на текущем шаге
        sub_x, sub_y = sample_train_data(X, Y, batch_size)
        # вычисляем градиент и значение функции потерь
        w_grad, cur_loss = grad_fun(sub_x, sub_y, w)
        # изменяем веса с заданным шагом в направлении антиградиента
        w -= step * w_grad # w = w - step * grad

        # запоминаем лучшее решение
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_w = w
        if i%10 == 0: #выводим информацию о решении оптимизационной задачи на каждом 10-м шаге 
            print("iter = %04d\tloss = %7f\tbest_loss=%7f\f|w| = %7f\t|g| = %7f" % 
               (i, cur_loss, best_loss, np.linalg.norm(w), np.linalg.norm(w_grad)))
    
    # возвращаем набор параметров, соответствующий минимальному значению функции потреь
    return best_w

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    More details: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def analytic_grad_softmax(x, y, w, reg_loss_weight=0.01):
    # forward
    x = np.atleast_2d(x)
    scores = w.dot(x.T)
    exp_scores = np.exp(scores)
    prob = exp_scores[y] / exp_scores.sum()
    data_loss = -np.log(prob)
    reg_loss = np.sum(w * w)
    loss = data_loss + reg_loss_weight * reg_loss
    # backward
    dreg = reg_loss_weight * 2.0 * w
    dw0 = exp_scores / exp_scores.sum()
    dw1 = dw0.dot(x)
    dw1[y] = - x
    dw = dw1 + dreg
    return (dw, loss)

def analytic_grad_softmax_vec_dummy(X, Y, w, reg_loss_weight=0.01):
    dw = np.zeros(w.shape)
    loss = 0
    for x, y in zip(X, Y):
        local_dw, local_loss = analytic_grad_softmax(x, y, w, reg_loss_weight)
        dw += local_dw
        loss += local_loss
    return (dw, loss)

def analytic_grad_hinge(x, y, w, reg_loss_weight=0.01):
    #forward pass
    x = np.atleast_2d(x)
    scores = w.dot(x.T)
    margins = np.maximum(0.0, scores - scores[y] + 1.0)
    margins[y] = 0.0
    data_loss = margins.sum()
    reg_loss = np.sum(w * w)
    loss = data_loss + reg_loss_weight * reg_loss
    #backward pass
    dreg = reg_loss_weight * 2.0 * w
    dmax = (margins > 0.0).astype(np.float)
    dmax[y] = -1.0 * (np.sum(dmax))
    dw = dmax.dot(x) + dreg
    return (dw, loss)

def analytic_grad_hinge_vec(X, Y, w, reg_loss_weight=0.01):
    Y = label_to_one_hot(Y)
    scores = w.dot(X.T).T
    margins = np.array([np.maximum(0, s - s[yi] + 1) for s, yi in zip(scores, Y)])
    margins[Y] = 0
    data_loss = margins.sum()
    reg_loss = np.sum(w*w)
    loss = data_loss + reg_loss_weight * reg_loss
    dreg = len(Y) * reg_loss_weight * 2.0 * w
    dmax = (margins > 0.0).astype(np.float)
    dmax[Y] = -1.0 * dmax.sum(1)
    dw = dmax.T.dot(X) + dreg
    return (dw, loss)

def analytic_grad_hinge_vec_dummy(X, Y, w, reg_loss_weight=0.01):
    dw = np.zeros(w.shape)
    loss = 0
    for x, y in zip(X, Y):
        local_dw, local_loss = analytic_grad_hinge(x, y, w, reg_loss_weight)
        dw += local_dw
        loss += local_loss
    return (dw, loss)

def test_analytic_and_numeric():
    for _ in range(100):
        n = np.random.randint(2, 12)
        c = np.random.randint(2, 13)
        x = np.random.randn(1, n)
        y = np.random.randint(c)
        w = np.random.randn(c, n)
        ag = analytic_grad_hinge(x, y, w)
        ng = numeric_gradient(x, y, w, hinge_loss)
        if not np.allclose(ag[0], ng[0], atol=0.001, rtol=0.001) or \
        not np.isclose(ag[1],ng[1],atol=0.001, rtol=0.001):
            print("Error!: ")
            print(ag)
            print(ng)
            break


print("Linear classification example")

# загрузка выборки
wine = datasets.load_wine()

# вывод информации о выборке
print(wine.DESCR)

X_raw, y = wine.data, wine.target

# нормализация признаков: вычитаем среднее и делим на вариацию
X_norm = (X_raw - X_raw.mean(0))/X_raw.std(0)

# добавим константу к признакам
X = np.hstack((X_norm, np.ones((X_norm.shape[0],1))))

# разделим выборку на обучающую и тестовую
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33, random_state=42)

# для целей текущего эксперимента зададим функцию потерь по всей выборке через hinge_loss
# с L2-регуляризацией
reg_coef = 1.e-3
global_hinge_loss = lambda X, Y, w : global_loss(X, Y, w, hinge_loss)

# для вычисления градиента будем использовать численную оценку
delta = 1.e-6 # шаг для оценки частных производных
numeric_gradient_hinge = lambda X, Y, w: numeric_gradient(X, Y, w, global_hinge_loss, delta)

# определим характеристики нашей задачи
num_samples, num_features = X_train.shape
num_classes = len(set(y_train))
num_weights = num_features * num_classes

print("Количество неизвестных параметров модели: %d" % num_weights)
print("Размер выборки: %d" % num_samples)

# инициализируем параметры линейной модели случайными нормальными (Гауссовскими) величинами
# с нормировкой по количесту параметров модели
w = np.random.randn(num_classes, num_features) / num_weights + 1.e-3

# сделаем 500 итераций с шагом по градиенту 1.e-3
#w = sgd(X_train, y_train, w, analytic_grad_hinge_vec, # numeric_gradient_hinge,
#        batch_size=32, n_iter=1000, step=1.e-3)

w = sgd(X_train, y_train, w, analytic_grad_softmax_vec_dummy, # numeric_gradient_softmax,
        batch_size=32, n_iter=1000, step=1.e-3)

# сделаем ещё 100 итераций с шагом по градиенту 1.e-4 для более точной подгонки
#w = sgd(X_train, y_train, w, analytic_grad_hinge_vec, #numeric_gradient_hinge,
#        batch_size=32, n_iter=1000, step=1.e-4)

w = sgd(X_train, y_train, w, analytic_grad_softmax_vec_dummy, # numeric_gradient_softmax,
        batch_size=32, n_iter=1000, step=1.e-4)

# построим предсказание с помощю обученной модели:
predict_train = w.dot(X_train.T).argmax(0)
predict_test = w.dot(X_test.T).argmax(0)

# оценим качество полученной модели на обучающей и тестовой выборках:
confusion_mat_train = metrics.confusion_matrix(y_train, predict_train)
confusion_mat_test = metrics.confusion_matrix(y_test, predict_test)

acc_train = metrics.accuracy_score(y_train, predict_train)
acc_test = metrics.accuracy_score(y_test, predict_test)

plt.figure()
plot_confusion_matrix(confusion_mat_train, classes=wine.target_names,
                  title='Confusion matrix on train, acc = %.3f' % acc_train)

plt.figure()
plot_confusion_matrix(confusion_mat_train, classes=wine.target_names, normalize=True,
                  title='Confusion matrix on train, acc = %.3f' % acc_train)

plt.figure()
plot_confusion_matrix(confusion_mat_test, classes=wine.target_names,
                  title='Confusion matrix on test, acc = %.3f' % acc_test)

plt.figure()
plot_confusion_matrix(confusion_mat_test, classes=wine.target_names, normalize=True,
                  title='Confusion matrix on test, acc = %.3f' % acc_test)

plt.show()

np.log(3)
print(w)
