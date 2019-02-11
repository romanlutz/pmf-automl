import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.impute
import time
import torch
import kernels
import gaussian_process_latent_variable_model
from utils import transform_forward, transform_backward
import bayesian_optimization

torch.set_default_tensor_type(torch.FloatTensor)

fn_data = 'all_normalized_accuracy_with_pipelineID.csv'
fn_train_ix = 'ids_train.csv'
fn_test_ix = 'ids_test.csv'
fn_data_feats = 'data_feats_featurized.csv'

def get_data():
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bayesian_optimization as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """

    df = pd.read_csv(fn_data)
    pipeline_ids = df['Unnamed: 0'].tolist()
    dataset_ids = df.columns.tolist()[1:]
    dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    Y = df.values[:,1:].astype(np.float64)

    ids_train = np.loadtxt(fn_train_ix).astype(int).tolist()
    ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ytrain = Y[:, ix_train]
    Ytest = Y[:, ix_test]

    df = pd.read_csv(fn_data_feats)
    dataset_ids = df[df.columns[0]].tolist()

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]

    Ftrain = df.values[ix_train, 1:]
    Ftest = df.values[ix_test, 1:]

    return Ytrain, Ytest, Ftrain, Ftest

def train(model, optimizer, f_callback=None, f_stop=None):

    iteration = 0
    while True:

        try:
            t = time.time()

            # set gradients of all model parameters to 0
            optimizer.zero_grad()
            # using __call__ of our model calls the forward function
            negative_log_likelihood = model()
            # calculate gradient
            negative_log_likelihood.backward()
            # perform parameter update based on current gradient
            optimizer.step()

            iteration += 1
            t = time.time() - t

            if f_callback is not None:
                f_callback(model, negative_log_likelihood, iteration, t)

            # f_stop should not be a substantial portion of total iteration time
            if f_stop is not None and f_stop(model, negative_log_likelihood, iteration, t):
                break

        except KeyboardInterrupt:
            break

    return model

def bayesian_optimization_search(model, bo_n_init, bo_n_iterations, Ytrain, Ftrain, ftest, ytest, do_print=False):
    """
    Initializes BayesianOptimization with L1 warm-start (using dataset features). Returns a
    numpy array of length bo_n_iterations holding the best performance attained
    so far per iteration (including initialization).

    bo_n_iterations includes initialization iterations, i.e., after warm-start, BayesianOptimization
    will run for bo_n_iterations - bo_n_init iterations.
    """

    predictions = bayesian_optimization.BayesianOptimization(model.dim, model.kernel, bayesian_optimization.expected_improvement,
                  variance=transform_forward(model.variance))
    ix_evaluated = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []

    def _process_ix(ix, predictions, model, ytest, ix_evaluated, ix_candidates):
        predictions.add(model.X[ix], ytest[ix])
        ix_evaluated.append(ix)
        ix_candidates.remove(ix)

    def _print_status(ix, bo_iteration, ytest, ybest, do_print):
        if do_print:
            print('Iteration: %d, %g [%d], Best: %g' % (bo_iteration, ytest[ix], ix, ybest))

    ix_init = bayesian_optimization.init_l1(Ytrain, Ftrain, ftest).tolist()
    for bo_iteration in range(bo_n_init):
        ix = ix_init[bo_iteration]
        if not np.isnan(ytest[ix]):
            _process_ix(ix, predictions, model, ytest, ix_evaluated, ix_candidates)
        ybest = predictions.ybest
        if ybest is None:
            ybest = np.nan
        ybest_list.append(ybest)

        _print_status(ix, bo_iteration, ytest, ybest, do_print)

    for bo_iteration in range(bo_n_init, bo_n_iterations):
        ix = ix_candidates[predictions.next(model.X[ix_candidates])]
        _process_ix(ix, predictions, model, ytest, ix_evaluated, ix_candidates)
        ybest = predictions.ybest
        ybest_list.append(ybest)

        _print_status(ix, bo_iteration, ytest, ybest, do_print)

    return np.asarray(ybest_list)

def random_search(bo_n_iterations, ytest, speed=1, do_print=False):
    """
    Speed denotes how many random queries are performed per iteration.
    """

    ix_evaluated = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    ybest = np.nan

    for bo_iteration in range(bo_n_iterations):
        for _ in range(speed):
            ix = ix_candidates[np.random.permutation(len(ix_candidates))[0]]
            if np.isnan(ybest):
                ybest = ytest[ix]
            else:
                if ytest[ix] > ybest:
                    ybest = ytest[ix]
            ix_evaluated.append(ix)
            ix_candidates.remove(ix)
        ybest_list.append(ybest)

        if do_print:
            print('Iteration: %d, %g [%d], Best: %g' % (bo_iteration, ytest[ix], ix, ybest))

    return np.asarray(ybest_list)

if __name__ == '__main__':
    # TODO: make these specifiable through command line params
    # train and evaluation settings
    Q = 20  # number of latent dimensions
    batch_size = 50  # size of dataset batches
    n_epochs = 300
    lr = 1e-7  # called eta in the paper
    N_max = 1000
    bo_n_init = 5
    bo_n_iterations = 200
    save_checkpoint = False
    fn_checkpoint = None
    checkpoint_period = 50

    # train
    Ytrain, Ytest, Ftrain, Ftest = get_data()
    max_iterations = int(Ytrain.shape[1] / batch_size * n_epochs)

    def f_stop(model, negative_log_likelihood, iteration, t):

        if iteration >= max_iterations - 1:
            print('max_iterations (%d) reached' % max_iterations)
            return True

        return False

    variances = []
    log_probabilities = []
    t_list = []
    def f_callback(model, negative_log_likelihood, iteration, t):
        variances.append(transform_forward(model.variance).item())
        log_probabilities.append(model().item()/model.D)
        if iteration == 1:
            t_list.append(t)
        else:
            t_list.append(t_list[-1] + t)

        if save_checkpoint and not (iteration % checkpoint_period):
            torch.save(model.state_dict(), fn_checkpoint + '_it%d.pt' % iteration)

        print('iteration=%d, log probability=%g, variance=%g, t: %g'
              % (iteration, log_probabilities[-1], transform_forward(model.variance), t_list[-1]))

    # create initial latent space with PCA, first imputing missing observations
    imputer = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    X = sklearn.decomposition.PCA(Q).fit_transform(imputer.fit(Ytrain).transform(Ytrain))

    # define model
    kernel = kernels.Add(kernels.RBF(Q, lengthscale=None), kernels.White(Q))
    model = gaussian_process_latent_variable_model.GaussianProcessLatentVariableModel(Q, X, Ytrain, kernel, N_max=N_max, D_max=batch_size)
    if save_checkpoint:
        torch.save(model.state_dict(), fn_checkpoint + '_it%d.pt' % 0)

    # optimize
    print('training...')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model = train(model, optimizer, f_callback=f_callback, f_stop=f_stop)
    if save_checkpoint:
        torch.save(model.state_dict(), fn_checkpoint + '_itFinal.pt')

    # evaluate model and random baselines
    print('evaluating...')
    with torch.no_grad():
        Ytest = Ytest.astype(np.float32)

        regrets_automl = np.zeros((bo_n_iterations, Ytest.shape[1]))
        regrets_random1x = np.zeros((bo_n_iterations, Ytest.shape[1]))
        regrets_random2x = np.zeros((bo_n_iterations, Ytest.shape[1]))
        regrets_random4x = np.zeros((bo_n_iterations, Ytest.shape[1]))

        for d in np.arange(Ytest.shape[1]):
            print(d)
            ybest = np.nanmax(Ytest[:,d])
            regrets_random1x[:,d] = ybest - random_search(bo_n_iterations,
                                                          Ytest[:,d], speed=1)
            regrets_random2x[:,d] = ybest - random_search(bo_n_iterations,
                                                          Ytest[:,d], speed=2)
            regrets_random4x[:,d] = ybest - random_search(bo_n_iterations,
                                                          Ytest[:,d], speed=4)
            regrets_automl[:,d] = ybest - bayesian_optimization_search(model, bo_n_init, bo_n_iterations,
                                                    Ytrain, Ftrain, Ftest[d,:],
                                                    Ytest[:,d])

        results = {
            'pmf': regrets_automl,
            'random1x': regrets_random1x,
            'random2x': regrets_random2x,
            'random4x': regrets_random4x
        }
