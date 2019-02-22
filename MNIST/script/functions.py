import numpy as np
import torch

def calc_energy(h, encoder, decoder, x, beta):
    """
     Calculate energy U and gradient of U with respect to h
    """
    h = h.clone().detach().requires_grad_(True)
    log_QhGx = encoder.calc_logQhGx(x, h)
    log_Pxh = decoder.calc_logPxh(x,h)
    energy = - (beta * log_Pxh + (1-beta)*log_QhGx)    
    energy.backward(torch.ones_like(energy))
    
    return energy, h.grad, log_QhGx.data, log_Pxh.data
    
def HMC(current_q, encoder, decoder, epsilon, L, x, beta):
    ''' Hamiltonian Monte Carlo (HMC) Algorithm
    
    '''
    ## sample a new momentum
    current_q.requires_grad_(False)
    current_p = torch.randn_like(current_q)
    
    #### update momentum and position ####
    ## proposed (q,p)
    q = current_q.clone().detach()
    p = current_p.clone().detach()        

    # ## step size
    # epsilon = epsilon.reshape(-1, 1)

    ## propogate state (q,p) using Hamiltonian equation with
    ## leapfrog integration
    
    ## propagate momentum by a half step at the beginning
    U, grad_U, _, _ = calc_energy(q, encoder, decoder, x, beta)
    p = p - 0.5*epsilon*grad_U

    ## propagate position and momentum alternatively by a full step
    ## L is the number of steps
    for i in range(L):
        q = q + epsilon * p
        U, grad_U, _, _ = calc_energy(q, encoder, decoder, x, beta)
        if i != L-1:
            p = p - epsilon*grad_U

    ## propagate momentum by a half step at the end
    p = p - 0.5*epsilon*grad_U

    ## calculate Hamiltonian of current state and proposed state
    current_U, _, _, _ = calc_energy(current_q, encoder, decoder, x, beta)
    current_K = torch.sum(0.5*current_p**2, -1)
    current_E = current_U + current_K
    
    proposed_U, _, _, _ = calc_energy(q, encoder, decoder, x, beta)
    proposed_K = torch.sum(0.5*p**2, -1)
    proposed_E = proposed_U + proposed_K

    ## accept proposed state using Metropolis criterion
    flag_accept = torch.rand_like(proposed_E) <= torch.exp(-(proposed_E - current_E))
    current_q[flag_accept] = q[flag_accept]
    
    return flag_accept, current_q


# def calc_energy(z, parameters):
#     """
#     Calculate energy U and gradient of U with respect to z

#     Args:
#         z: tensor of size batch_size x d, where d is the dimension of z
#     """

#     ## make a copy of z
#     z = z.clone().detach().requires_grad_(True)
#     d = z.shape[-1]     ## dimension of z
    
#     pi = z.new_tensor(np.pi, requires_grad = False)

#     ## parameters for the Gaussian mixture distribution and reference
#     ## Gaussian distribution
#     mu_0 = parameters['mu_0']
#     sigma_0 = parameters['sigma_0']
#     mu_1 = parameters['mu_1']
#     sigma_1 = parameters['sigma_1']
#     mu_r = parameters['mu_r']
#     sigma_r = parameters['sigma_r']

#     ## inverse temperature
#     beta = parameters['beta']    

    
#     sigma_0_product = torch.exp(torch.sum(torch.log(sigma_0)))
#     sigma_1_product = torch.exp(torch.sum(torch.log(sigma_1)))    

#     ## energy of the target distribution: -logP(z)
#     energy_target = -torch.log(
#         0.3*1.0/((2*pi)**(d/2.0)*sigma_0_product) * \
#         torch.exp(torch.sum(-0.5*((z - mu_0)/sigma_0)**2, -1)) + \
#         0.7*1.0/((2*pi)**(d/2.0)*sigma_1_product) * \
#         torch.exp(torch.sum(-0.5*((z - mu_1)/sigma_1)**2, -1))
#     )

#     ## energy of the reference distribution: -logQ(z)
#     energy_ref = 0.5*d*torch.log(2*pi) + torch.sum(torch.log(sigma_r)) + \
#                  torch.sum(0.5*((z - mu_r)/sigma_r)**2, -1)

#     ## energy of the intermediate distribution:
#     energy = beta*energy_target + (1-beta)*energy_ref

#     ## use backpropgation to calculate force on z
#     energy.backward(torch.ones_like(energy))

#     return energy.data, z.grad, energy_target.data, energy_ref.data
