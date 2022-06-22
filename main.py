import seaborn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


def show_histogram(column_data, column):
    plt.hist(column_data)
    plt.xlabel(column)
    plt.ylabel('count')
    plt.show()


def show_all_historgams(wine_data_train):
    show_histogram(wine_data_train['free sulfur dioxide'], 'free sulfur dioxide')
    show_histogram(wine_data_train['alcohol'], 'alcohol')
    show_histogram(wine_data_train['chlorides'], 'chlorides')
    show_histogram(wine_data_train['citric acid'], 'citric acid')
    show_histogram(wine_data_train['density'], 'density')
    show_histogram(wine_data_train['fixed acidity'], 'fixed acidity')
    show_histogram(wine_data_train['pH'], 'pH')
    show_histogram(wine_data_train['residual sugar'], 'residual sugar')
    show_histogram(wine_data_train['sulphates'], 'sulphates')
    show_histogram(wine_data_train['total sulfur dioxide'], 'total sulfur dioxide')
    show_histogram(wine_data_train['type'], 'type')
    show_histogram(wine_data_train['volatile acidity'], 'volatile acidity')


def print_describe(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data.describe())


def manage_null_values(wine_data, train_data):
    null_columns = wine_data.columns[wine_data.isnull().any()]
    print(wine_data[null_columns].isnull().sum())

    # null replace with mean
    for i in null_columns:
        wine_data[i] = wine_data[i].fillna(train_data[i].mean())  # mode()[0]

    return wine_data


def load_data(wine_data, train_data):
    print_describe(wine_data)

    wine_data = manage_null_values(wine_data, train_data)
    print_describe(wine_data)

    return wine_data


def min_max_normalization(wine_data):
    # Min-Max Normalization

    scaler = preprocessing.MinMaxScaler()
    names = wine_data.columns
    d = scaler.fit_transform(wine_data)
    scaled_data = pd.DataFrame(d, columns=names)

    print(scaled_data.head())

    print_describe(scaled_data)

    return scaled_data


def logist_reg(scaled_data, wine_data_quality):
    lg = LogisticRegression(random_state=0, max_iter=1000).fit(scaled_data, wine_data_quality)
    return lg


def logistic_regression(scaled_data_train, wine_data_train_quality, scaled_data_test, wine_data_test_quality):
    # Logisticka regresia
    data_log_reg = logist_reg(scaled_data_train, wine_data_train_quality.values.ravel())
    predict_log_reg = data_log_reg.predict(scaled_data_test)
    report_log_reg = metrics.classification_report(wine_data_test_quality, predict_log_reg)
    print(report_log_reg)


def define_model(scaled_data_train):
    # define the keras model
    model = Sequential()
    model.add(Dense(8, input_shape=(scaled_data_train.shape[1],), activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def show_plot(training, report_type):
    plt.plot(training.history[report_type], label='testovacie')
    plt.plot(training.history['val_'+report_type], label='validacne')
    plt.title(report_type)
    plt.legend()
    plt.show()


def plot_cf(cf):
    plot_confusion_matrix(conf_mat=cf, figsize=(11, 8))
    plt.xlabel('Predikcia', fontsize=18)
    plt.ylabel('Očakávanie', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def neural_network(scaled_data_train, wine_data_train_quality, scaled_data_test, wine_data_test_quality):
    model = define_model(scaled_data_train)
    optimizer = Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    x_train1, x_valid, y_train1, y_valid = train_test_split(scaled_data_train, wine_data_train_quality, test_size=0.20,
                                                            random_state=42)
    print('Dlzka trenovacich dat: ', len(x_train1))
    print('Dlzka validacnych dat: ', len(x_valid))
    print('Dlzka testovacich dat: ', len(scaled_data_test))

    # training = model.fit(x_train1, y_train1, epochs=300, verbose=0)

    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    training = model.fit(x_train1, y_train1, epochs=300, verbose=0, validation_data=(x_valid, y_valid), callbacks=[es])

    print(training.history.keys())
    predict_neuron = (pd.DataFrame(model.predict(scaled_data_test), columns=['quality'])).round()
    report_neuron = metrics.classification_report(wine_data_test_quality, predict_neuron)
    print(report_neuron)

    # evaluate the model
    _, train_acc = model.evaluate(x_train1, y_train1, verbose=0)
    _, test_acc = model.evaluate(x_valid, y_valid, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    show_plot(training, 'loss')
    show_plot(training, 'accuracy')

    cf = (confusion_matrix(wine_data_test_quality, predict_neuron))
    plot_cf(cf)



if __name__ == '__main__':
    pd.set_option('display.width', 360)

    wine_data_train = pd.read_csv("data/wine_train.csv")
    wine_data_test = pd.read_csv("data/wine_test.csv")

    # show_all_historgams(wine_data_train)

    wine_data_train = load_data(wine_data_train, wine_data_train)
    wine_data_train_quality = wine_data_train['quality']
    wine_data_train = wine_data_train[wine_data_train.columns.difference(["quality"])]

    scaled_data_train = min_max_normalization(wine_data_train)

    show_histogram(wine_data_train['alcohol'], 'alcohol')
    show_histogram(scaled_data_train['alcohol'], 'alcohol')

    wine_data_test = load_data(wine_data_test, wine_data_train)
    wine_data_test_quality = wine_data_test['quality']
    wine_data_test = wine_data_test[wine_data_test.columns.difference(["quality"])]

    scaled_data_test = min_max_normalization(wine_data_test)

    logistic_regression(scaled_data_train, wine_data_train_quality, scaled_data_test, wine_data_test_quality)

    neural_network(scaled_data_train, wine_data_train_quality, scaled_data_test, wine_data_test_quality)
