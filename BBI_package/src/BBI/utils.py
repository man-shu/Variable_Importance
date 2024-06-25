import numpy as np
from sklearn.preprocessing import StandardScaler


def create_X_y(
    X,
    y,
    bootstrap=True,
    split_perc=0.8,
    prob_type="regression",
    list_cont=None,
    random_state=None,
):
    """Create train/valid split of input data X and target variable y.
    Parameters
    ----------
    X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The input samples before the splitting process.
    y: ndarray, shape (n_samples, )
        The output samples before the splitting process.
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set.
    split_perc: float, default=0.8
        The training/validation cut for the provided data.
    prob_type: str, default='regression'
        A classification or a regression problem.
    list_cont: list, default=[]
        The list of continuous variables.
    random_state: int, default=2023
        Fixing the seeds of the random generator.

    Returns
    -------
    X_train_scaled: {array-like, sparse matrix}, shape (n_train_samples, n_features)
        The bootstrapped training input samples with scaled continuous variables.
    y_train_scaled: {array-like}, shape (n_train_samples, )
        The bootstrapped training output samples scaled if continous.
    X_valid_scaled: {array-like, sparse matrix}, shape (n_valid_samples, n_features)
        The validation input samples with scaled continuous variables.
    y_valid_scaled: {array-like}, shape (n_valid_samples, )
        The validation output samples scaled if continous.
    X_scaled: {array-like, sparse matrix}, shape (n_samples, n_features)
        The original input samples with scaled continuous variables.
    y_valid: {array-like}, shape (n_samples, )
        The original output samples with validation indices.
    scaler_x: scikit-learn StandardScaler
        The standard scaler encoder for the continuous variables of the input.
    scaler_y: scikit-learn StandardScaler
        The standard scaler encoder for the output if continuous.
    valid_ind: list
        The list of indices of the validation set.
    """
    rng = np.random.RandomState(random_state)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    n = X.shape[0]

    if bootstrap:
        train_ind = rng.choice(n, n, replace=True)
    else:
        train_ind = rng.choice(
            n, size=int(np.floor(split_perc * n)), replace=False
        )
    valid_ind = np.array([ind for ind in range(n) if ind not in train_ind])

    X_train, X_valid = X[train_ind], X[valid_ind]
    y_train, y_valid = y[train_ind], y[valid_ind]

    # Scaling X and y
    X_train_scaled = X_train.copy()
    X_valid_scaled = X_valid.copy()
    X_scaled = X.copy()

    if len(list_cont) > 0:
        X_train_scaled[:, list_cont] = scaler_x.fit_transform(
            X_train[:, list_cont]
        )
        X_valid_scaled[:, list_cont] = scaler_x.transform(
            X_valid[:, list_cont]
        )
        X_scaled[:, list_cont] = scaler_x.transform(X[:, list_cont])
    if prob_type == "regression":
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_valid_scaled = scaler_y.transform(y_valid)
    else:
        y_train_scaled = y_train.copy()
        y_valid_scaled = y_valid.copy()

    return (
        X_train_scaled,
        y_train_scaled,
        X_valid_scaled,
        y_valid_scaled,
        X_scaled,
        y_valid,
        scaler_x,
        scaler_y,
        valid_ind,
    )


def convert_predict_proba(list_probs):
    """If the classification is done using a one-hot encoded variable, the list of
    probabilites will be a list of lists for the probabilities of each of the categories.
    This function takes the probabilities of having each category (=1 with binary) and stack
    them into one ndarray.
    """
    if len(list_probs.shape) == 3:
        list_probs = np.array(list_probs)[..., 1].T
    return list_probs
