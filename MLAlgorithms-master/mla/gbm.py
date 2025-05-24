import sys, os

# assumir que o script está em MLI-Practical-Assignment, no mesmo nível de MLAlgorithms-master
base = os.path.dirname(__file__)              # .../MLI-Practical-Assignment
mla_path = os.path.join(base, "MLAlgorithms-master")
sys.path.insert(0, mla_path)

# coding:utf-8
import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from zipfile import ZipFile

"""
References:
https://arxiv.org/pdf/1603.02754v3.pdf
http://www.saedsayad.com/docs/xgboost.pdf
https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
http://stats.stackexchange.com/questions/202858/loss-function-approximation-with-taylor-expansion
"""
def train_test_split(X, y, test_size=0.3, random_state=None, stratify=None):
    """
    Divide arrays X e y em treino e teste.
    Se stratify for fornecido, faz divisão estratificada segundo y.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n = X.shape[0]
    indices = np.arange(n)
    if stratify is not None:
        # division estratificada para cada classe em stratify
        train_idx = []
        test_idx = []
        classes, counts = np.unique(stratify, return_counts=True)
        for cls, cnt in zip(classes, counts):
            cls_idx = indices[stratify == cls]
            np.random.shuffle(cls_idx)
            n_test = int(np.floor(test_size * cnt))
            test_idx.extend(cls_idx[:n_test].tolist())
            train_idx.extend(cls_idx[n_test:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        # divisão aleatória simples
        np.random.shuffle(indices)
        n_test = int(np.floor(test_size * n))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def recall_score(y_true, y_pred, pos_label=1):
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def precision_score(y_true, y_pred, pos_label=1):
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def f1_score(y_true, y_pred, pos_label=1):
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def roc_auc_score(y_true, y_score):
    # assume binário com rótulos 0 e 1
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return 0.0
    tp = 0
    fp = 0
    tps = []
    fps = []
    for yi in y_sorted:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)
    tps = np.array(tps) / P
    fps = np.array(fps) / N
    # adicionar (0,0) e (1,1)
    tpr = np.concatenate(([0], tps, [1]))
    fpr = np.concatenate(([0], fps, [1]))
    # regra trapezoidal
    return np.trapz(tpr, fpr)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ------------------------------------------------------------
# Implementação de Gradient Boosting baseada em MLAlgorithms
# (com pré-processamento de numéricas e categóricas)
# ------------------------------------------------------------
from mla.base import BaseEstimator
from mla.ensemble.base import mse_criterion
from mla.ensemble.tree import Tree

class Loss:
    def __init__(self, regularization=1.0):
        self.regularization = regularization

    def grad(self, actual, predicted):
        raise NotImplementedError()

    def hess(self, actual, predicted):
        raise NotImplementedError()

    def approximate(self, actual, predicted):
        g = self.grad(actual, predicted).sum()
        h = self.hess(actual, predicted).sum() + self.regularization
        return g / h

    def transform(self, pred):
        return pred

    def gain(self, actual, predicted):
        g = self.grad(actual, predicted).sum()
        h = self.hess(actual, predicted).sum() + self.regularization
        return 0.5 * (g * g / h)

class LeastSquaresLoss(Loss):
    def grad(self, actual, predicted):
        return actual - predicted
    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogisticLoss(Loss):
    def grad(self, actual, predicted):
        return actual * expit(-actual * predicted)
    def hess(self, actual, predicted):
        p = expit(predicted)
        return p * (1 - p)
    def transform(self, output):
        return expit(output)

class GradientBoosting(BaseEstimator):
    def __init__(self,
                 n_estimators,
                 learning_rate=0.1,
                 max_features=10,
                 max_depth=2,
                 min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = None
        self.trees = []
        self._feature_names = None
        self._num_cols = []
        self._cat_cols = []
        self._means = None
        self._stds = None
        self._cat_uniques = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            is_num = X.dtypes.apply(pd.api.types.is_numeric_dtype)
            self._num_cols = [c for c, flag in zip(self._feature_names, is_num) if flag]
            self._cat_cols = [c for c, flag in zip(self._feature_names, is_num) if not flag]
            self._means = X[self._num_cols].mean()
            self._stds = X[self._num_cols].std(ddof=0).replace({0: 1.0})
            self._cat_uniques = {c: list(X[c].astype(str).unique())
                                  for c in self._cat_cols}
            X_proc = self._preprocess(X)
        else:
            X_proc = X.astype(np.float32)

        self._setup_input(X_proc, y)
        self.y_mean = np.mean(y) if y is not None else 0.0
        self._train()

    def _preprocess(self, X):
        # normalização z-score
        if self._num_cols:
            X_num = (X[self._num_cols] - self._means) / self._stds
        else:
            X_num = pd.DataFrame(index=X.index)
        # one-hot
        cat_parts = []
        for c in self._cat_cols:
            col = X[c].astype(str)
            cats = self._cat_uniques[c]
            df_one = pd.DataFrame(
                0,
                index=X.index,
                columns=[f"{c}__{v}" for v in cats]
            )
            for v in cats:
                df_one.loc[col == v, f"{c}__{v}"] = 1
            cat_parts.append(df_one)
        X_tot = pd.concat([X_num] + cat_parts, axis=1) if cat_parts else X_num
        return X_tot.values.astype(np.float32)

    def _train(self):
        y_pred = np.zeros(self.n_samples, dtype=np.float32)
        for _ in range(self.n_estimators):
            residuals = self.loss.grad(self.y, y_pred)
            tree = Tree(regression=True, criterion=mse_criterion)
            targets = {"y": residuals, "actual": self.y, "y_pred": y_pred}
            tree.train(self.X, targets,
                       max_features=self.max_features,
                       min_samples_split=self.min_samples_split,
                       max_depth=self.max_depth,
                       loss=self.loss)
            y_pred += self.learning_rate * tree.predict(self.X)
            self.trees.append(tree)

    def _predict_raw(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.float32)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and self._feature_names is not None:
            X_proc = self._preprocess(X)
        else:
            X_proc = X.astype(np.float32)
        raw = self._predict_raw(X_proc)
        return self.loss.transform(raw)

class GradientBoostingRegressor(GradientBoosting):
    def fit(self, X, y=None):
        self.loss = LeastSquaresLoss()
        super().fit(X, y)

class GradientBoostingClassifier(GradientBoosting):
    def fit(self, X, y=None):
        y2 = (y * 2) - 1
        self.loss = LogisticLoss()
        super().fit(X, y2)
    def predict(self, X):
        return super().predict(X)

# ------------------------------------------------------------
# Função de carregamento de dataset
# ------------------------------------------------------------
def load_dataset(path, target_col):
    """
    Carrega CSV direto ou de dentro de um ZIP.
    Retorna (X_df, y_array).
    """
    if path.lower().endswith('.zip'):
        with ZipFile(path) as zf:
            fname = next(n for n in zf.namelist() if n.endswith('.csv'))
            df = pd.read_csv(zf.open(fname))
    else:
        df = pd.read_csv(path)
    if target_col not in df:
        raise ValueError(f"target column '{target_col}' not found")
    X = df.drop(target_col, axis=1)
    y = df[target_col].values
    return X, y

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GradientBoosting (classifier or regressor) sem sklearn"
    )
    parser.add_argument("-d", "--dataset", required=True,
                        help="Caminho para .csv ou .zip com um .csv")
    parser.add_argument("-t", "--target", required=True,
                        help="Nome da coluna de target")
    parser.add_argument("-m", "--mode", choices=["classification", "regression"],
                        default="classification",
                        help="Modo: classification ou regression")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Fracão do dataset para teste")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Semente aleatória")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Número de árvores")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Profundidade máxima das árvores")
    parser.add_argument("--min-samples-split", type=int, default=10,
                        help="Min amostras para split")

    args = parser.parse_args()

    # Carregar
    X, y = load_dataset(args.dataset, args.target)

    # Dividir
    strat = y if args.mode == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X.values if isinstance(X, pd.DataFrame) else X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=strat
    )

    # Instanciar
    if args.mode == "classification":
        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split
        )

    # Treinar
    model.fit(X_train, y_train)

    # Prever e avaliar
    print(f"\n=== Results ({args.mode}) ===")
    if args.mode == "classification":
        y_proba = model.predict(X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        print("accuracy:", accuracy_score(y_test, y_pred))
        print("recall (minority):", recall_score(y_test, y_pred, pos_label=1))
        print("f1 score:", f1_score(y_test, y_pred))
        print("auc roc:", roc_auc_score(y_test, y_proba))
    else:
        y_pred = model.predict(X_test)
        print("mse:", mean_squared_error(y_test, y_pred))
        print("mae:", mean_absolute_error(y_test, y_pred))
        print("r2:", r2_score(y_test, y_pred))