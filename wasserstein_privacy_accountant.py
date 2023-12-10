
import torch 
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import hyp1f1
import numpy as np





class WassersteinPrivacyAccountant:
	def __init__(self, order, steps_total, beta):
		"""
		Creating WassersteinPrivacyAccountant Framework

		Parameters:
		-----------
		orders: list or int
			The orders set for Wasserstein differential privacy.
		steps_total:
			Total iterations steps in Wasserstein differential privacy accountant.
		

		steps_accounted:
			Total accounted steps in Wasserstein differential privacy accountant.

		privacy_loss: 
			The privacy_loss of Wasserstein differential privacy.

		"""
		self.order = order
		self.steps_total = steps_total
		self.beta = beta


		self.privacy_loss = torch.zeros_like(torch.tensor(self.order), dtype = torch.float)
		self.history = []
		self.steps_accounted = 0


	def __str__(self):
		return "WassersteinPrivacyAccountant"




	def get_privacy(self, target_privacy=None, target_delta=None):

		"""
		Parameters:
		-----------
		target_privacy: float
			Target privacy budget
		target_failure:  float
			Taget failure probability


		Returns:
		--------
			A tuple (epsilon, delta)

		"""

		if (target_privacy is None) and (target_delta is None):
			raise ValueError("At least one of the two parameters should be defined.")
		if (target_privacy is not None) and (target_delta is not None):
			raise ValueError("Only one of the two parameters can be assigned to obtain the other.")
		if target_privacy is None:
			return torch.min(self.privacy_loss - np.log(target_delta)/self.beta).item(), target_delta
			#return torch.min(self.privacy_loss/self.beta), target_delta
		else:
			return target_privacy, self.beta * torch.exp(self.privacy_loss - target_privacy)


	def accumulate(self, lgrad, rgrad, scale_param, subsampling_rate, steps=1):
		"""
            Computing accumulative privacy loss under WDP.
            
            Parameters
            ----------
            lgrad : tensor, required
                One of the neiboring gradient tensor.
            rgrad : tensor, required
                Another of the neiboring gradient tensor.
            scale_param : float
            	Scale parameter of Gaussian distribution.
            subsampling_rate : float, required
                Sampling rate in DP-SGD.
    
            
            Returns
            -------
            out : tensor
                Accumulative privacy loss under WDP.
        """

		wdp = self.compute_privacy_loss(
									lgrad = lgrad, 
									rgrad = rgrad,
									scale_param = scale_param,
									subsampling_rate = subsampling_rate,
									)

		self.privacy_loss += wdp
		self.history.append(self.privacy_loss)
		self.steps_accounted += steps
		return self.history[-1]



	def compute_privacy_loss(self, lgrad, rgrad, scale_param, subsampling_rate):
		"""
            Computing privacy loss under WDP.
            
            Parameters
            ----------
            lgrad : tensor, required
                One of the neiboring gradient tensor.
            rgrad : tensor, required
                Another of the neiboring gradient tensor.
            scale_param : float
            	Scale parameter of Gaussian distribution.
            subsampling_rate : float, required
                Sampling rate in DP-SGD.
    
            
            Returns
            -------
            out : tensor
                Privacy loss under WDP.
        """


		abs_moment = self.compute_absolute_moments(
										lgrad = lgrad, 
										rgrad = rgrad,
										scale_param = scale_param,
										subsampling_rate = subsampling_rate,
										)
		sum_moment = torch.sum(abs_moment)

		privacy_loss = torch.pow(sum_moment, 1/self.order)

		# print("====privacy_loss====", privacy_loss)

		return torch.min(privacy_loss).item()



	def compute_absolute_moments(self, lgrad, rgrad, scale_param, subsampling_rate):
		"""
            Computing the absolute moments of a Gaussian distribution.
            
            Parameters
            ----------
            lgrad : tensor, required
                One of the neiboring gradient tensor.
            rgrad : tensor, required
                Another of the neiboring gradient tensor.

            scale_param : float
            	Scale parameter of Gaussian distribution.
            subsampling_rate : float, required
                Sampling rate in DP-SGD.
    
            
            Returns
            -------
            out : tensor
                The absolute moments of a Gaussian distribution.
        """
		
		if subsampling_rate > 1 or subsampling_rate < 0:
			raise ValueError("subsampling_rate should smaller than 1 and larger than 0.")

		coeff_of_val = (2 - 2*subsampling_rate + 2*pow(subsampling_rate, 2))
		sigma_square = pow(scale_param, 2)

		variance = coeff_of_val * sigma_square


		first_term = pow(2 * variance, self.order / 2)

		gamma_term = self.compute_gamma_function() / (np.sqrt(np.pi))

		grad_dist = self.compute_grad_norm(lgrad, rgrad)

		kummor_term = self.compute_kummor_confluent_hypergeometric_function(
					subsampling_rate = subsampling_rate, 
					grad_distance = grad_dist, 
					variance = variance, 
				)

		abs_moment = first_term * gamma_term * kummor_term

		# print('=*'*10, torch.min(abs_moment))

		return abs_moment



	def compute_gamma_function(self):
		z_param = (self.order + 1)/2 
		return gamma(z_param)



	def compute_kummor_confluent_hypergeometric_function(self, subsampling_rate, grad_distance, variance):
		"""
            Computing the norm of neighboring gradients.
            
            Parameters
            ----------
            subsampling_rate : float, required
                Sampling rate in DP-SGD.
            grad_distance : tensor, required
                The norm of neighboring gradients.
            variance : float
            	The variance of Gaussian noise in DP-SGD.
            
            Returns
            -------
            out : tensor
                A tensor represents the norm of the input neiboring gradients.
        """

		a_param = -self.order / 2

		b_param = 1/2
		
		q_square = pow(subsampling_rate, 2)
		d_t_square = torch.pow(grad_distance, 2)

		x_param = - (q_square * d_t_square) / (2 * variance)

		kummor = hyp1f1(a_param, b_param, x_param.cpu())

		return kummor



	def compute_grad_norm(self, lgrad, rgrad):
		"""
            Computing the norm of neighboring gradients.
            
            Parameters
            ----------
            lgrad : tensor, required
                One of the neiboring gradient tensor.
            rgrad : tensor, required
                Another of the neiboring gradient tensor.
            
            Returns
            -------
            out : tensor
                A tensor represents the norm of the input neiboring gradients.
        """

		grad_distance = lgrad - rgrad

		if not torch.is_tensor(grad_distance):
			grad_distance = torch.tensor(grad_distance)


		grad_distance = torch.norm(grad_distance, p=2, dim=-1).view(-1)

		# print('='*10, grad_distance.shape)

		return grad_distance






		