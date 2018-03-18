import numpy as np


def weights_to_string(ws, is_int=False):
    ws = np.array(ws)
    ws[np.abs(ws) < 1e-10] = 0
    ws = ws.flatten().tolist()
    s = ''
    for w in ws:
        s = s + ('%.3f,' if not is_int else '%d,') % w
    return s


def test_wrapper(fn, n_lines):
    np.random.seed(42)
    try:
        res = fn()
    except Exception as e:
        res = ['ERROR @' + fn.__name__ + '\t' + str(e), ] * n_lines

    return res


def test_mse():
    from utils import mean_squared_error

    result = []
    x = np.linspace(-1, 1, num=20).reshape(-1, 1)
    y_true = x * x
    y_pred = x * x - 2 * x

    mse = mean_squared_error(y_true.flatten().tolist(),
                             y_pred.flatten().tolist())
    result.append(mse)
    result.append(
        mean_squared_error(
            np.random.normal(size=(10,)).flatten().tolist(),
            np.random.normal(size=(10,)).flatten().tolist()
        )
    )
    return ['[TEST mean_squared_error],' + weights_to_string(result)]


def test_f1_score():
    from utils import f1_score

    result = []
    y_true = np.random.randint(low=0, high=4, size=(100))
    y_true[y_true <= 2] = 0
    y_true[y_true > 2] = 1
    y_pred = np.linspace(0, 1, num=100)
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    score = f1_score(y_true.flatten().tolist(), y_pred.flatten().tolist())
    result.append(score)

    result.append(f1_score(
        np.random.randint(0, high=2, size=(100,)).flatten().tolist(),
        np.random.randint(0, high=2, size=(100,)).flatten().tolist()
    ))

    return ['[TEST f1_score],' + weights_to_string(result)]


def test_polynomial_features():
    from utils import polynomial_features

    x = np.random.random((3, 2)).tolist()
    result = []
    for k in [2, 3]:
        result.append('[TEST polynomial_features],' +
                      weights_to_string(polynomial_features(x, k=k)))
    return result


def test_euclidean_distance():
    from utils import euclidean_distance

    result = []
    for i in range(1, 3):
        x = np.linspace(-1, 1, num=i * 5).reshape(-1, 1)
        y = (x * x - i * x).flatten().tolist()
        x = np.sin(x).flatten().tolist()
        result.append('[TEST euclidean_distance],' +
                      weights_to_string([euclidean_distance(x, y)]))
    return result


def test_inner_product_distance():
    from utils import inner_product_distance

    result = []
    for i in range(1, 3):
        x = np.linspace(-1, 1, num=i * 5).reshape(-1, 1)
        y = (x * x - i * x).flatten().tolist()
        x = np.sin(x).flatten().tolist()
        result.append('[TEST inner_product_distance],' +
                      weights_to_string([inner_product_distance(x, y)]))
    return result


def test_gaussian_kernel_distance():
    from utils import gaussian_kernel_distance
    result = []
    for i in range(1, 3):
        x = np.linspace(-1, 1, num=i * 5).reshape(-1, 1)
        y = (x * x - i * x).flatten().tolist()
        x = np.sin(x).flatten().tolist()
        result.append('[TEST gaussian_kernel_distance],' +
                      weights_to_string([gaussian_kernel_distance(x, y)]))
    return result


def test_NormalizationScaler():
    from utils import NormalizationScaler

    result = []
    x = np.random.random(size=(3, 3)).tolist()
    x_test = np.random.random(size=(3, 3)).tolist()

    scaler = NormalizationScaler()
    result.append('[TEST NormalizationScaler],' + weights_to_string(scaler(x)))
    result.append('[TEST NormalizationScaler],' +
                  weights_to_string(scaler(x_test)))

    return result


# test MinMaxScaler
def test_MinMaxScaler():
    from utils import MinMaxScaler

    result = []
    x = np.random.random(size=(3, 3)).tolist()
    x_test = np.random.random(size=(3, 3)).tolist()

    scaler = MinMaxScaler()
    result.append('[TEST MinMaxScaler],' + weights_to_string(scaler(x)))
    result.append('[TEST MinMaxScaler],' + weights_to_string(scaler(x_test)))

    return result


# test Linear Regression & Ridge Regression
def test_linear_regression():
    from hw1_lr import LinearRegression

    result = []

    x = np.linspace(-2, 2, num=5).reshape(-1, 1)
    y = x * x * x

    x = x.tolist()
    y = y.flatten().tolist()

    model = LinearRegression(nb_features=1)
    model.train(x, y)
    result.append('[TEST LinearRegression]' +
                  weights_to_string(model.get_weights()))
    result.append('[TEST LinearRegression]' +
                  weights_to_string(model.predict(x)))

    return result


def test_linear_regression2():
    from hw1_lr import LinearRegression

    result = []

    x = np.array([[1], [-1]])
    y = x * 3 + 2

    x = x.tolist()
    y = y.flatten().tolist()

    model = LinearRegression(nb_features=1)
    model.train(x, y)
    result.append('[TEST LinearRegression]' +
                  weights_to_string(model.get_weights()))
    result.append('[TEST LinearRegression]' +
                  weights_to_string(model.predict(x)))

    return result


def test_ridge_regression():
    from hw1_lr import LinearRegressionWithL2Loss

    result = []
    x = np.linspace(-2, 2, num=5).reshape(-1, 1)
    x = np.hstack([x, x ** 2, np.sin(x)])
    y = 0.3 * x[:, 0] - 1.5 * x[:, 1] + 3 * x[:, 2] - 0.01

    x = x.tolist()
    y = y.flatten().tolist()

    for alpha in [0, 0.1, 2]:
        model = LinearRegressionWithL2Loss(nb_features=1, alpha=alpha)
        model.train(x, y)
        result.append('[TEST LinearRegressionWithL2Loss]' +
                      weights_to_string(model.get_weights()))
        result.append('[TEST LinearRegressionWithL2Loss]' +
                      weights_to_string(model.predict(x)))

    return result


def test_knn():
    from hw1_knn import KNN
    from utils import euclidean_distance

    result = []
    x = np.random.normal(size=(100, 2)).tolist()
    x = set([tuple(_) for _ in x])
    x = list([list(_) for _ in x])
    y = np.random.randint(low=0, high=2, size=(50)).flatten().tolist()

    x_test = x[50:]
    x = x[:50]

    for k in [3, 7, 11]:
        model = KNN(k=k, distance_function=euclidean_distance)
        model.train(x, y)
        result.append('[TEST KNN2],' +
                      weights_to_string(model.predict(x_test), is_int=True))
    return result


def test_knn2():
    from hw1_knn import KNN
    from utils import euclidean_distance

    result = []
    x = np.random.normal(size=(100, 2)).tolist()
    x = set([tuple(_) for _ in x])
    x = list([list(_) for _ in x])
    y = np.random.randint(low=0, high=5, size=(50)).flatten().tolist()

    x_test = x[50:]
    x = x[:50]

    for k in [1]:
        model = KNN(k=k, distance_function=euclidean_distance)
        model.train(x, y)
        result.append('[TEST KNN2],' +
                      weights_to_string(model.predict(x_test), is_int=True))
    return result


def test_perceptron():
    from hw1_perceptron import Perceptron
    from data import generate_data_perceptron

    result = []
    x, y = generate_data_perceptron(3)

    # Not passing in margin, so if they handle margin it will be 1e-4 else 0
    model = Perceptron(3)
    convergence = model.train(x, y)
    acc = np.sum(np.array(y) == np.array(model.predict(x))) / len(y)

    result.append('[TEST Perceptron seperable convergence],' +
                  str(convergence) + ' ')
    result.append('[TEST Perceptron seperable accuracy is 100],' +
                  str(acc == 1.0) + ' ')

    x, y = generate_data_perceptron(3, seperation=1)

    # Not passing in margin, so if they handle margin it will be 1e-4 else 0
    model = Perceptron(3, margin=0)
    convergence = model.train(x, y)
    acc = np.sum(np.array(y) == np.array(model.predict(x))) / len(y)
    result.append('[TEST Perceptron non-seperable convergence],' +
                  str(convergence) + ' ')
    result.append('[TEST Perceptron non-seperable accuracy is not 100],' +
                  str(acc < 1.0) + ' ')
    return result


if __name__ == '__main__':
    result = []

    # LR: 14 lines
    result += test_wrapper(test_polynomial_features, 2)
    result += test_wrapper(test_mse, 1)
    result += test_wrapper(test_f1_score, 1)
    result += test_wrapper(test_linear_regression, 2)
    result += test_wrapper(test_linear_regression2, 2)
    result += test_wrapper(test_ridge_regression, 2 * 3)

    # kNN: 14 lines
    result += test_wrapper(test_euclidean_distance, 2)
    result += test_wrapper(test_inner_product_distance, 2)
    result += test_wrapper(test_gaussian_kernel_distance, 2)
    result += test_wrapper(test_NormalizationScaler, 2)
    result += test_wrapper(test_MinMaxScaler, 2)
    result += test_wrapper(test_knn, 3)
    result += test_wrapper(test_knn2, 1)

    # Perceptron: 4 lines
    result += test_wrapper(test_perceptron, 4)

    with open('output_hw1.csv', 'w') as f:
        for line in result:
            f.write(line[:-1] + '\n')
