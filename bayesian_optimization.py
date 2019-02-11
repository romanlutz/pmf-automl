import torch
import gaussian_process_latent_variable_model
import numpy as np
import scipy.stats as st

def expected_improvement(mean, variance, ybest, xi=0.01, eps=1e-12):
    '''
    xi is a parameter to encourage exploration
    '''
    standard_deviation = torch.sqrt(variance) + eps
    gamma = (mean - ybest - xi)/standard_deviation

    return standard_deviation * (torch.distributions.Normal(0,1).cdf(gamma) * gamma
                                 + torch.distributions.Normal(0,1).log_prob(gamma).exp())

def init_l1(Ytrain, Ftrain, ftest, n_init=5):

    distance = np.abs(Ftrain - ftest).sum(axis=1)
    ix_closest = np.argsort(distance)[:n_init]
    ix_nonnan_pipelines = np.where(np.invert(np.isnan(Ytrain[:, ix_closest].sum(axis=1))))[0]
    ranks = np.apply_along_axis(st.rankdata, 0, Ytrain[ix_nonnan_pipelines[:, None], ix_closest])
    average_pipeline_ranks = ranks.mean(axis=1)
    ix_init = ix_nonnan_pipelines[np.argsort(average_pipeline_ranks)[::-1]]

    return ix_init[:n_init]

class BayesianOptimization(gaussian_process_latent_variable_model.GaussianProcess):

    def __init__(self, dim, kernel, acquisition_function, **kwargs):
        super(BayesianOptimization, self).__init__(dim, np.asarray([]), np.asarray([]), kernel,
                                 **kwargs)

        self.acquisition_function = acquisition_function
        self.ybest = None
        self.xbest = None

    def add(self, xnew, ynew):
        '''
        Extends X with xnew and y with ynew through concatenation.
        '''
        xnew_ = torch.tensor(xnew, dtype=self.X.dtype).reshape((1,-1))
        self.X = torch.cat((self.X, xnew_))
        ynew_ = torch.tensor([ynew], dtype=self.y.dtype)
        self.y = torch.cat((self.y, ynew_))
        if self.ybest is None or ynew_ > self.ybest:
            self.ybest = ynew_
            self.xbest = xnew_
        self.N += 1

    def next(self, Xcandidates):
        '''
        Selects the next candidate using the acquisition function.
        Depending on the acquisition function this can factor in uncertainty
        about the predicted performance.
        '''
        if not self.N:
            return torch.randperm(Xcandidates.size()[0])[0]

        mean, variance = self.posterior(Xcandidates)

        return torch.argmax(self.acquisition_function(mean, variance, self.ybest))
