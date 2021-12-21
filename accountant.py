import ot
import numpy as np
import scipy.special as spec

class Accounting(object):
    def __init__(self, param1, param2, order, delta):
        self.param1 = param1
        self.param2 = param2
        self.order = order
        self.delta = delta
        #self.generate_param()
        #self.Wasserstein_iteration()


    def Wasserstein_iteration(self):
        WasserD = ot.wasserstein_1d(x_a=spec.softmax(self.param1), x_b=spec.softmax(self.param2), p=self.order)
        epsilon_i_bound = WasserD 
        return epsilon_i_bound

    def Wasserstein_accounting(self):
        composition += self.Wasserstein_iteration()

    def Bayesian_iteration(self, T, lambda_):
        shoulder = T * lambda_ * self.Renyi_divergence()
        cost_t = np.log(np.mean(np.exp(shoulder)))
        cost_t = np.power(cost_t, 1/T)
        return cost_t

    def Renyi_divergence(self):
        if self.order == 1:
            out = np.sum(self.param1 * np.log(self.param1 / self.param2))
        elif self.order != 1:
            expect = np.sum(self.param2* ((self.param1/self.param2) ** self.order))
            out = 1/(self.order - 1) * np.log(expect)
        return out

    def Bayesian_accounting(T, lambda_):
        iter_part = 0
        iter_part += self.Bayesian_accounting(T, lambda_)
        return 1/lambda_ * iter_part - 1/lambda_ * np.log(self.delta)

    def moment_iteration(self):
        cost_t = np.log(spec.softmax(self.param1)) - np.log(spec.softmax(self.param2))
        alpha_moment = np.exp(self.order * cost_t)
        alpha = np.log(np.mean(alpha_moment))
        return alpha

    def moment_accounting(self, lambda_):
        alpha_summa = 0
        alpha_summa += self.Moment_iteration()
        lambda_times_epsilon = np.min(alpha_summa - np.log(self.delta))
        return lambda_times_epsilon/lambda_



