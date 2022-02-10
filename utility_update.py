import numpy as np
import scipy
import copy
import scipy.io as scio
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from scipy.stats import poisson, norm, gamma, dirichlet, uniform, beta
from scipy.special import gammaln
import math
# from numba import jit
#
import pyximport
pyximport.install()
from cython_file import update_D, update_C, cal_C_lower, sample_z_cython, CRT_sample, cal_C_lower_0, cal_C_lower_together,update_C_integer, update_D_integer, simple_sum, CRT_sample_binomial, CRT_sample_vector, positive_poisson_sample_cyuse

# def CRT_sample(n_customer, alpha_val):
#     return np.sum(uniform.rvs(size = n_customer) < alpha_val/(alpha_val+np.arange(n_customer)))




def load_data_fan(fileName):

    # read data
    relation_matrix = scio.loadmat(fileName)['datas'].astype(int)
    # data process, change to 0/1, Rii = 0(not consider)
    relation_matrix[relation_matrix>1] = 1
    relation_matrix[relation_matrix<0] = 0
    relation_matrix[(np.arange(relation_matrix.shape[0]), np.arange(relation_matrix.shape[0]))] = 0
    # separate data: # test_matrix= test, relation_matrix= training, 相互的位置为-1


    [data_num, col_num] = (relation_matrix.shape)
    test_matrix = np.ones(relation_matrix.shape)*(-1)
    test_ratio = 0.1
    for ii in range(data_num):
        # use False, choose different number
        test_index_i = np.random.choice(col_num, int(col_num*test_ratio), replace=False)
        test_matrix[ii, test_index_i] = copy.copy(relation_matrix[ii, test_index_i])
        relation_matrix[ii, test_index_i] = -1

    nonzero_judge = ((np.sum(relation_matrix==1, axis=0)+np.sum(relation_matrix==1, axis=1))>0)
    relation_matrix = relation_matrix[nonzero_judge]
    relation_matrix = relation_matrix[:, nonzero_judge]

    test_matrix = test_matrix[nonzero_judge]
    test_matrix = test_matrix[:, nonzero_judge]

    data_num = len(relation_matrix)

    return relation_matrix, data_num, test_matrix


def positive_poisson_sample(z_lambda):
    # return positive truncated poisson random variables Z = 1, 2, 3, 4, ...
    # z_lambda: parameter for Poisson distribution

    candidate = 1000 #1000
    can_val = np.arange(1, candidate)
    log_vals = can_val*np.log(z_lambda)-np.cumsum(np.log(can_val))
    vals = np.exp(log_vals - np.max(log_vals))

    a = np.random.multinomial(1, pvals=(vals/np.sum(vals)))
    select_i = np.where(a == 1)[0][0]

    # select_val = np.random.choice(can_val, p = (vals/np.sum(vals)))
    return can_val[select_i]


def calcualteAUC(y_true, y_scores):
    n1 = np.sum(y_true)
    no = len(y_true) - n1
    rank_indcs =np.argsort(y_scores)
    R_sorted = y_true[rank_indcs]
    #+1 because indices in matlab begins with 1
    # #however in python, begins with 0
    So=np.sum(np.where(R_sorted>0)[0]+1)
    aucValue = float(So - (n1*(n1+1))/2)/(n1*no)
    return aucValue


def initialize_model(dataR, dataNum, KK, LL, case_number, bounded_version):
    # Input:
    # dataR: positive relational data # positive edges x 2 ,dataR[ij][0] relates to dataR[ij][1]
    # KK: number of communities
    # LL: number of layers

    # Output:
    # M[-1]: Poisson distribution parameter in generating X_{ik}
    # X_i: latent counts for node i
    # Z_ik: latent integers summary, calculating as \sum_{j,k_2} Z_{ij,kk_2}
    # Z_k1k2: latent integers summary, calculating as \sum_{k,k_2} Z_{ij,kk_2}
    # pis: (LL+1) X N X KK: layer-wise mixed-membership distributions
    # Lambdas: community compatibility matrix
    # connections: information propagation index: node connections[ij, 0] would propagate information to node connections[ij, 1]
    # C_val: (LL) X (number of positive edges) X K: counts value of information propagation from ll-layer to (ll+1)-th layer
    #   C_val needs to viewed together with connections
    #   C_val[ll, ij, k] refers to propagate node connections[ij, 0] to node connections[ij, 1] at latent feature k from l-th layer to (l+1)-th layer
    # D_val: (LL) X N X K X K: counts value of information propagation from ll-layer to (ll+1)-th layer
    # W_val: N X N: information propagation matrix, each row of W_val sums to 1
    # Psi_val: K X K: information propagation matrix, each row of Psi_val sums to 1

    #### all ij(Rij=1) + all ii(Rii)
    connections = np.vstack((dataR, (np.arange(dataNum)[:, np.newaxis]*np.ones((1, 2))).astype(int)))
    # (np.arange(dataNum)[:, np.newaxis]*np.ones((1, 2))).astype(int)=array([[   0,    0],[   1,    1],...[3311, 3311]]) diagnose location
    # np.vstack, put two array as one array

    #### pis, C_val, D_val, W_val, Psi_val
    pis = np.zeros((LL + 1, dataNum, KK))

    # M = np.ones(LL+1)*dataNum
    M = np.ones(LL + 1) * 1000 # 1000

    ## pis[0]
    pis_inte = gamma.rvs(a=1*np.ones((dataNum, KK))+1e-16, scale=1) + 1e-16 # same as dirichlet(alpha=1)
    pis[0] = pis_inte/(np.sum(pis_inte, axis=1)[:, np.newaxis])

    ## pis[1:], C_val, D_val, W_val, Psi_val
    # W_val, Psi_val
    W_val = np.zeros((dataNum, dataNum))
    Psi_val = np.zeros((KK, KK))
    for kk in range(KK):
        Psi_val[kk] = dirichlet.rvs(np.ones(KK))
    for ii in range(dataNum):
        # the list of i' who has relationship with ii, including ii itself. so it wont be ([])
        from_ii_nodes = connections[connections[:, 0]==ii, 1]
        # only update the W_val[ii, i'] has the relationship with ii
        W_val[ii, from_ii_nodes] = dirichlet.rvs(np.ones(len(from_ii_nodes)))

    # C_val[ll]_ii'k~Poisson(M(ll)*pis[ll]_ik*wii') size of L*((Rii'=1)+Rii)*K; -----> from pis[ll] to pis[ll+1]
    # D_val[ll]_ii'k~Poisson(M(ll)*pis[ll]_ik*Psi_kk') size of L*K*K; -----> from pis[ll] to pis[ll+1]
    C_val = np.zeros((LL, len(connections), KK))
    D_val = np.zeros((LL, dataNum, KK, KK))
    for ll in range(LL):
        C_val[ll] = poisson.rvs(M[ll]*pis[ll][connections[:, 0]]*W_val[connections[:, 0], connections[:, 1]][:, np.newaxis]) # length*k * length*1 = length*k
        # for ij in range(len(connections)):
        #     C_val[ll, ij] = poisson.rvs(M[ll]*pis[ll][connections[ij, 0]]*W_val[connections[ij, 0], connections[ij, 1]])
        for ii in range(dataNum):
            D_val[ll, ii] = poisson.rvs(M[ll] * pis[ll,ii][:, np.newaxis] * Psi_val)  # k*1 * k* k' = k*k'
            # Ri'i, for i, sum the i'
            pis_inte = gamma.rvs(a=(1 + np.sum(C_val[ll, connections[:, 1]==ii], axis=0) + np.sum(D_val[ll,ii], axis=0) + 1e-16), scale=1)+1e-16
            pis[ll+1, ii] = pis_inte / np.sum(pis_inte)
            # pis[ll+1, ii] = dirichlet.rvs(np.sum(C_val[ll, connections[:, 1]==ii], axis=0)+1)

    #### X_i~Poisson(M(L)*pis[LL]), Lambda, Z_ik, Z_k1k2
    X_i = poisson.rvs(M[-1]*pis[LL]).astype(float)

    R_KK = np.ones((KK, KK))/(KK ** 2)
    np.fill_diagonal(R_KK, 1/KK)
    Lambdas = gamma.rvs(a=R_KK, scale=1)

    Z_ik = np.zeros((dataNum, KK), dtype=int) # N*K
    Z_k1k2 = np.zeros((KK, KK), dtype=int) # K*K  Zij_K1K2
    for ii in range(len(dataR)): # here only consider Rij=1 and i!=j  list of each pair of ij
        pois_lambda = (X_i[dataR[ii][0]][:, np.newaxis] * X_i[dataR[ii][1]][np.newaxis, :]) * Lambdas
        # X_i(K1*1)*Xj(K2*1)*Lambdas(K1*K2)=K1*K2   the p in multinomial
        total_val = positive_poisson_sample(np.sum(pois_lambda))
        # total_val = Zij,..# positive poisson the number of rolling in multinomial
        new_counts = np.random.multinomial(total_val, pois_lambda.reshape((-1)) / np.sum(pois_lambda)).reshape((KK, KK))
        # {Zij_{k1,k2}}k1,k2   k1*k2
        Z_k1k2 += new_counts # sum ij in Zij or Zji
        Z_ik[dataR[ii][0]] += np.sum(new_counts, axis=1)
        Z_ik[dataR[ii][1]] += np.sum(new_counts, axis=0)
        # sum all j in Zij or Zji
    return M, X_i, Lambdas, Z_ik, Z_k1k2, pis, C_val, D_val, W_val, Psi_val, connections



class UDBN_class:
    def __init__(self, connections, dataNum, LL, KK, M, X_i, Lambdas, Z_ik, Z_k1k2, pis, C_val, D_val, W_val, Psi_val):

        self.dataNum = dataNum
        self.LL = LL
        self.KK = KK

        self.connection = connections ##  LE x 2, records the linkage connection between nodes. From nodes connections[:, 0] to nodes connections[:, 1]

        self.X_i = X_i
        self.Lambdas = Lambdas
        self.M_c = M[:(-1)]
        self.M_d = M[:(-1)]
        self.M_X = M[0]/(self.KK**2)
        self.Z_ik = Z_ik
        self.Z_k1k2 = Z_k1k2.astype(np.intc)
        self.alpha_pi = np.ones((LL+1, KK))*0.1
        self.alpha_phi = np.ones((LL, KK))*0.1
        self.alpha_omega = np.ones((LL, dataNum))*0.1

        self.xi_omega = np.ones(LL)
        self.xi_phi = np.ones(LL)

        # self.alpha_val = 1

        self.W_val = [W_val for _ in range(LL)]
        self.Psi_val = [Psi_val for _ in range(LL)]
        self.D_val = D_val
        self.C_val = C_val # (LL) X LE X K, counting information between the layers
        self.pis = pis  #(LL+1) X N X K

        self.infinite_indi = False # defines whether we are using finite or infinite number of latent features

        self.r_k = np.ones(self.KK)
        self.epsilon = 1

        self.beta_lambda = 1

        self.nodes_from_ii_list = []
        self.nodes_to_ii_list = []

        self.nodes_from_ii_index = []
        self.nodes_to_ii_index = []

        for ii in range(self.dataNum):
            self.nodes_from_ii_index.append(np.where(self.connection[:, 0] == ii)[0].tolist())
            self.nodes_from_ii_list.append(connections[self.nodes_from_ii_index[-1], 1])

            self.nodes_to_ii_index.append(np.where(self.connection[:, 1] == ii)[0])
            self.nodes_to_ii_list.append(connections[self.nodes_to_ii_index[-1], 0])

    # def sample_w_psi(self, case_number):
    #     # sample the information propagation matrix
    #     for ll in range(self.LL):
    #         if (case_number == 1) | (case_number == 3):
    #             counts_from_ii = np.zeros((self.dataNum, self.dataNum))
    #             h_ii = np.zeros((self.dataNum, self.dataNum))
    #             log_q_i = np.zeros(self.dataNum)
    #
    #             for ii in range(self.dataNum):
    #
    #                 counts_from_ii[ii][self.nodes_from_ii_list[ii]] = np.sum(self.C_val[ll, self.nodes_from_ii_index[ii]], axis=1)
    #                 alpha_omega_ii = copy.deepcopy(self.alpha_omega[ll][self.nodes_from_ii_list[ii]])
    #                 alpha_omega_ii[self.nodes_from_ii_list[ii]==ii] *= self.xi_omega[ll]
    #                 log_q_i[ii] = np.log((1e-16+beta.rvs(a=np.sum(alpha_omega_ii), b=np.sum(counts_from_ii[ii])))/(1+1e-16))
    #                 h_ii[ii, self.nodes_from_ii_list[ii]] = CRT_sample_vector(counts_from_ii[ii][self.nodes_from_ii_list[ii]].astype(np.intc),alpha_omega_ii)
    #
    #             for ii in range(self.dataNum):
    #                 log_q_i_to_ii = copy.copy(log_q_i[self.nodes_to_ii_list[ii]])
    #                 log_q_i_to_ii[self.nodes_to_ii_list[ii]==ii] *= self.xi_omega[ll]
    #                 self.alpha_omega[ll][ii] = gamma.rvs(a=0.1 + np.sum(h_ii[:, ii]), scale=1) / (0.1 - np.sum(log_q_i_to_ii))
    #
    #             self.xi_omega[ll] = gamma.rvs(a = 0.1 + np.sum(np.diag(h_ii)), scale = 1)/(0.1-np.sum(self.alpha_omega[ll]*log_q_i))
    #
    #             for ii in range(self.dataNum):
    #
    #                 alpha_omega_ii = self.alpha_omega[ll][self.nodes_from_ii_list[ii]]
    #                 alpha_omega_ii[self.nodes_from_ii_list[ii]==ii] *= self.xi_omega[ll]
    #                 inte_val = gamma.rvs(a =alpha_omega_ii+counts_from_ii[ii][self.nodes_from_ii_list[ii]], scale = 1) # alpha_w = 1
    #                 self.W_val[ll][ii, self.nodes_from_ii_list[ii]] = inte_val/np.sum(inte_val)
    #
    #         if (case_number == 2) | (case_number == 3):
    #             counts_from_kk = np.sum(self.D_val[ll], axis=0)
    #
    #             q_k = beta.rvs(a=np.sum(self.alpha_phi[ll]) ,b=np.sum(counts_from_kk, axis=1))
    #             h_kk = CRT_sample(counts_from_kk.astype(np.intc),
    #                                        np.ones((self.KK, 1)).dot(self.alpha_phi[ll].reshape((1, -1))))
    #             self.alpha_phi[ll] = gamma.rvs(a=0.1 + np.sum(h_kk, axis=0), scale=1) / (0.1 - np.sum(np.log(q_k)))
    #
    #             inte_val = gamma.rvs(a=self.alpha_phi[ll][np.newaxis, :]+counts_from_kk, scale= 1) # alpha_Psi = 1
    #             self.Psi_val[ll] = inte_val/(np.sum(inte_val, axis=1)[:, np.newaxis])
    #

    def sample_w_psi(self, case_number):
        # sample the information propagation matrix
        for ll in range(self.LL):
            if (case_number == 1) | (case_number == 3):

                h_ii = np.zeros((self.dataNum, self.dataNum))
                log_q_i = np.zeros(self.dataNum)

                for ii in range(self.dataNum):
                    counts_from_ii = np.sum(self.C_val[ll, self.nodes_from_ii_index[ii]], axis=1)

                    log_q_i[ii] = np.log((1e-16+beta.rvs(a=len(self.nodes_from_ii_list[ii])*self.alpha_omega[ll][ii], b=np.sum(counts_from_ii)+1e-6))/(1+1e-16))
                    h_ii[ii, self.nodes_from_ii_list[ii]] = CRT_sample_vector(counts_from_ii.astype(np.intc),\
                                                                              np.ones(len(self.nodes_from_ii_list[ii]))*self.alpha_omega[ll][ii])

                    self.alpha_omega[ll][ii] = gamma.rvs(a=0.1 + np.sum(h_ii[ii]), scale=1) / (0.1 - len(self.nodes_from_ii_list[ii])*log_q_i[ii])

                    inte_val = gamma.rvs(a =self.alpha_omega[ll][ii]+counts_from_ii, scale = 1)+1e-6 # alpha_w = 1
                    self.W_val[ll][ii, self.nodes_from_ii_list[ii]] = inte_val/np.sum(inte_val)

            if (case_number == 2) | (case_number == 3):
                counts_from_kk = np.sum(self.D_val[ll], axis=0)

                q_k = beta.rvs(a=self.KK*self.alpha_phi[ll] ,b=np.sum(counts_from_kk, axis=1))
                h_kk = CRT_sample(counts_from_kk.astype(np.intc),
                                           (self.alpha_phi[ll].reshape((-1, 1)).dot(np.ones((1, self.KK)))))
                self.alpha_phi[ll] = gamma.rvs(a=0.1 + np.sum(h_kk, axis=1), scale=1) / (0.1 - self.KK*(np.log(q_k)))

                inte_val = gamma.rvs(a=self.alpha_phi[ll][:, np.newaxis]+counts_from_kk, scale= 1) # alpha_Psi = 1
                self.Psi_val[ll] = inte_val/(np.sum(inte_val, axis=1)[:, np.newaxis])


    # def sample_w_psi(self, case_number):
    #     # sample the information propagation matrix
    #     for ll in range(self.LL):
    #         if (case_number == 1) | (case_number == 3):
    #             counts_from_ii = np.zeros((self.dataNum, self.dataNum))
    #             h_ii = np.zeros((self.dataNum, self.dataNum))
    #             q_i = np.zeros(self.dataNum)
    #             for ii in range(self.dataNum):
    #                 judge_ii = (self.connection[:, 0] == ii)
    #                 counts_from_ii[ii][self.connection[judge_ii, 1]] = np.sum(self.C_val[ll, judge_ii], axis=1)
    #
    #                 q_i[ii] = (1e-16+beta.rvs(a=np.sum(self.alpha_omega[ll][self.connection[judge_ii, 1]]) ,b=np.sum(counts_from_ii[ii])))/(1+1e-16)
    #                 h_ii[ii, self.connection[judge_ii, 1]] = CRT_sample_vector(counts_from_ii[ii][self.connection[judge_ii, 1]].astype(np.intc),
    #                                        self.alpha_omega[ll][self.connection[judge_ii, 1]])
    #             self.alpha_omega[ll] = gamma.rvs(a=0.1 + np.sum(h_ii, axis=0), scale=1) / (0.1 - np.sum(np.log(q_i)))
    #
    #             for ii in range(self.dataNum):
    #                 judge_ii = (self.connection[:, 0] == ii)
    #                 inte_val = gamma.rvs(a =self.alpha_omega[ll][self.connection[judge_ii, 1]]+counts_from_ii[ii][self.connection[judge_ii, 1]], scale = 1) # alpha_w = 1
    #                 self.W_val[ll][ii, self.connection[judge_ii, 1]] = inte_val/np.sum(inte_val)

    def sample_X_i(self, dataR_matrix):

        # sample the latent counts X_i

        idx = (dataR_matrix != (-1))
        np.fill_diagonal(idx, False) # not need to consider i==j

        for nn in range(self.dataNum):
            Xik_Lambda = np.sum(np.dot(self.Lambdas, (idx[nn][:, np.newaxis] * self.X_i).T), axis=1) + \
                         np.sum(np.dot(self.Lambdas.T, (idx[:, nn][:, np.newaxis] * self.X_i).T), axis=1)
            log_alpha_X = np.log(self.M_X) + np.log(self.pis[-1][nn]) - Xik_Lambda
            for kk in range(self.KK):
                n_X = self.Z_ik[nn, kk]
                if n_X == 0:
                    select_val = poisson.rvs(np.exp(log_alpha_X[kk]))
                else:
                    candidates = np.arange(1, 1000 + 1)  # self.dataNum, we did not consider 0 because the ratio is 0 for sure
                    pseudos = candidates * log_alpha_X[kk] + n_X * np.log(candidates) - np.cumsum(np.log(candidates))
                    proportions = np.exp(pseudos - max(pseudos))
                    select_val = np.random.choice(candidates, p=proportions/np.sum(proportions))
                self.X_i[nn, kk] = select_val


    # def sample_Lambda_k1k2(self, dataR_matrix):
    #     # sample Lambda according to the gamma distribution
    #
    #     idx = (dataR_matrix != (-1))
    #     np.fill_diagonal(idx, False)
    #
    #     Phi_KK = np.dot(np.dot(self.X_i.T, idx), self.X_i)
    #     R_KK = np.ones((self.KK, self.KK))/(self.KK ** 2)
    #     np.fill_diagonal(R_KK, 1/self.KK)
    #     self.Lambdas = gamma.rvs(a=self.Z_k1k2 + R_KK, scale=1)/(1 + Phi_KK)

    def sample_Lambda_k1k2(self, dataR_matrix):
        idx = (dataR_matrix != (-1))
        np.fill_diagonal(idx, 0)
        Phi_KK = np.dot(np.dot(self.X_i.T, idx), self.X_i)
        # np.fill_diagonal(Phi_KK, np.diag(Phi_KK)/2)


        # sample r_k
        L_KK = np.zeros((self.KK, self.KK))

        R_KK = np.dot(self.r_k[:, np.newaxis], self.r_k[np.newaxis, :]).astype(np.double)
        np.fill_diagonal(R_KK, (self.r_k)*self.epsilon)

        p_kk_prime_one_minus = self.beta_lambda / (self.beta_lambda + Phi_KK)
        # for k1 in range(self.KK):
        #     for k2 in range(self.KK):
        #         if self.Z_k1k2[k1, k2]>0:
        #             L_KK[k1, k2] = CRT_sample(self.Z_k1k2[k1, k2], R_KK[k1, k2])
        #     add_val = np.sum(R_KK[k1]/self.r_k[k1]*np.log(p_kk_prime_one_minus[k1]))
        #     self.r_k[k1] = gamma.rvs(a = 1/self.KK+np.sum(L_KK[k1]), scale = 1)/(1-add_val)

        L_KK = CRT_sample(self.Z_k1k2.astype(np.intc), R_KK)
        for k1 in range(self.KK):
            add_val = np.sum(R_KK[k1]/self.r_k[k1]*np.log(p_kk_prime_one_minus[k1]))
            self.r_k[k1] = gamma.rvs(a = 1/self.KK+np.sum(L_KK[k1]), scale = 1)/(1-add_val)


        # sample epsilon
        add_val = np.sum(self.r_k * np.log(np.diag(p_kk_prime_one_minus)))
        self.epsilon = gamma.rvs(a = 1+np.sum(np.diag(L_KK)), scale = 1)/(1 - add_val)

        # sample lambda
        self.Lambdas = gamma.rvs(a = self.Z_k1k2 + R_KK, scale = 1)/(self.beta_lambda+Phi_KK)

        # sample beta
        self.beta_lambda = gamma.rvs(a = 1+np.sum(R_KK), scale = 1)/(1+np.sum(self.Lambdas))

        # # sample c_0_lambda
        # self.c_0_lambda = gamma.rvs(a=1+self.gamma_0_lambda, scale = 1)/(1+np.sum(self.r_k))

    def sample_Z_ik_k1k2(self, dataR):
        # sampling the latent integers

        Z_ik = np.zeros((self.dataNum, self.KK), dtype=int)
        Z_k1k2 = np.zeros((self.KK, self.KK), dtype=int)
        for ii in range(len(dataR)):
            pois_lambda = (self.X_i[dataR[ii][0]][:, np.newaxis]*self.X_i[dataR[ii][1]][np.newaxis, :])*self.Lambdas
            # total_val = positive_poisson_sample(np.sum(pois_lambda))
            total_val = positive_poisson_sample_cyuse(np.sum(pois_lambda))
            new_counts = np.random.multinomial(total_val, pois_lambda.reshape((-1))/np.sum(pois_lambda)).reshape((self.KK, self.KK))
            Z_k1k2 += new_counts
            Z_ik[dataR[ii][0]] += np.sum(new_counts, axis=1)
            Z_ik[dataR[ii][1]] += np.sum(new_counts, axis=0)

        # [Z_k1k2, Z_ik] = sample_z_cython(np.intc(len(dataR)), np.intc(self.KK), np.intc(self.dataNum), dataR.astype(np.intc), self.X_i, self.Lambdas)

        self.Z_k1k2 = Z_k1k2
        self.Z_ik = Z_ik

        # indexx = np.where(np.sum(Z_ik, axis=1)==0)[0]


    def sample_M(self):
        # updating the hyper-parameter M

        k_M = self.dataNum
        c_M = 1
        for ll in range(self.LL):
            self.M_c[ll] = gamma.rvs(a= k_M + np.sum(self.C_val[ll]), scale=1/(c_M + self.dataNum))

        for ll in range(self.LL):
            self.M_d[ll] = gamma.rvs(a= k_M + np.sum(self.D_val[ll]), scale=1/(c_M + self.dataNum))

        self.M_X = gamma.rvs(a= k_M/(self.KK**2) + np.sum(self.X_i), scale=1)/(c_M + self.dataNum)



    def sample_C_D_Pi(self, case_number, bounded_version):

        candidates = np.arange(1000).astype(np.intc) # 1000
        log_can = copy.copy(candidates)  ##### This is because log(0) is probamatic! We need to do some things before the error
        log_can[0] = 1
        denominator_log = np.cumsum(np.log(log_can))

        # if bounded_version == 1:
        #     alpha_k = 2.**(-np.arange(self.KK)-1)
        #     alpha_k[-1] = 2.**(-(self.KK-1))
        #     alpha_k_sum = 1.
        # else:
        #     alpha_k = np.ones(self.KK)
        #     alpha_k_sum = self.KK

        for ll in range(self.LL):

            if (case_number == 1)|(case_number==3):
                C_lower = cal_C_lower(np.intc(self.dataNum), np.intc(self.KK), self.connection.astype(np.intc), self.C_val[ll].astype(np.intc))
                C_higher = np.sum(C_lower,axis=1).astype(np.intc)
            else:
                C_lower = np.zeros((self.dataNum, self.KK)).astype(np.intc) # dataNum is for i'
                C_higher = np.zeros(self.dataNum).astype(np.intc)

            if (case_number == 2)|(case_number==3):
                D_lower = np.sum(self.D_val[ll], axis=1).astype(np.intc) # sum k' in D(l)_i'k'k
                D_higher = np.sum(D_lower, axis=1).astype(np.intc)
            else:
                D_lower = np.zeros((self.dataNum, self.KK)).astype(np.intc) # dataNum is for i'
                D_higher = np.zeros(self.dataNum).astype(np.intc)



            if (case_number == 1)|(case_number==3):
                self.C_val[ll] = update_C(np.intc(bounded_version), self.alpha_pi[ll+1], np.intc(self.KK), np.intc(len(candidates)), self.pis[ll], self.pis[ll+1], self.W_val[ll], self.M_c[ll], candidates, denominator_log, \
                                          self.C_val[ll].astype(np.intc), C_lower, C_higher, self.connection.astype(np.intc), D_lower,D_higher)
            else:
                self.C_val[ll] = np.zeros(self.C_val[ll].shape)

            if (case_number == 2) | (case_number == 3):
                self.D_val[ll] = update_D(np.intc(bounded_version), self.alpha_pi[ll+1], np.intc(self.dataNum), np.intc(self.KK), np.intc(len(candidates)), self.pis[ll], self.pis[ll+1], self.Psi_val[ll], self.M_d[ll], candidates, denominator_log,\
                         self.D_val[ll].astype(np.intc), D_lower, D_higher, C_lower, C_higher)
            else:
                self.D_val[ll] = np.zeros(self.D_val[ll].shape)


            countss_0 = np.zeros((self.dataNum, self.KK))
            countss_1 = np.zeros((self.dataNum, self.KK))
            if ll == 0:

                for ii in range(self.dataNum):
                    countss_0[ii] = np.sum(self.C_val[ll][self.connection[:, 0] == ii], axis=0) + np.sum(self.D_val[ll, ii], axis=1)

                try:
                    q_i = beta.rvs(a = np.sum(self.alpha_pi[ll]), b = np.sum(countss_0, axis=1)+1e-6)
                except:
                    a = 1
                h_ik = CRT_sample(countss_0.astype(np.intc), np.ones((self.dataNum, 1)).dot(self.alpha_pi[ll].reshape((1, -1))))

            elif  0 < ll < self.LL:

                 for ii in range(self.dataNum):
                     countss_1[ii] = np.sum(self.C_val[ll-1][self.connection[:, 1] == ii], axis=0) + np.sum(self.D_val[ll-1,ii], axis=0)
                     countss_0[ii] = np.sum(self.C_val[ll][self.connection[:, 0] == ii], axis=0) + np.sum(self.D_val[ll, ii], axis=1)

                 q_i = beta.rvs(a=np.sum(self.alpha_pi[ll])+np.sum(countss_1, axis=1), b=np.sum(countss_0, axis=1)+1e-6)
                 h_ik = CRT_sample_binomial(countss_0.astype(np.intc), np.ones((self.dataNum, 1)).dot(self.alpha_pi[ll].reshape((1, -1))), countss_1)

            self.alpha_pi[ll] = gamma.rvs(a=0.1 + np.sum(h_ik, axis=0), scale=1) / (0.1 - np.sum(np.log(q_i)))
            rdv = gamma.rvs(a=self.alpha_pi[ll]+countss_0 + countss_1 +1e-16, scale=1) + 1e-16
            self.pis[ll] = rdv/np.sum(rdv, axis=1)[:, np.newaxis]

            if ll == (self.LL-1):
                for ii in range(self.dataNum):
                     countss_1[ii] = np.sum(self.C_val[self.LL-1][self.connection[:, 1] == ii], axis=0) + np.sum(self.D_val[self.LL-1,ii], axis=0)
                countss_0 = self.X_i

                q_i = beta.rvs(a=np.sum(self.alpha_pi[self.LL])+np.sum(countss_1, axis=1), b=np.sum(countss_0, axis=1)+1e-6)
                h_ik = CRT_sample_binomial(countss_0.astype(np.intc), np.ones((self.dataNum, 1)).dot(self.alpha_pi[self.LL].reshape((1, -1))), countss_1)
                self.alpha_pi[self.LL] = gamma.rvs(a=0.1 + np.sum(h_ik, axis=0), scale=1) / (0.1 - np.sum(np.log(q_i)))

                rdv = gamma.rvs(a=self.alpha_pi[self.LL]+countss_0 + countss_1+1e-16, scale=1) + 1e-16
                self.pis[self.LL] = rdv/np.sum(rdv, axis=1)[:, np.newaxis]



#############################################################################################################
#############################################################################################################
##############################################################################################################

    def sample_C_D_integer_process(self, bounded_version):

        if bounded_version == 1:
            alpha_k = 2.**(-np.arange(self.KK)-1)
            alpha_k[-1] = 2.**(-(self.KK-1))
            alpha_k_sum = 1.
        else:
            alpha_k = np.ones(self.KK)
            alpha_k_sum = self.KK

        candidates = np.arange(1000).astype(int) # 1000
        log_can = copy.copy(candidates)  ##### This is because log(0) is probamatic! We need to do some things before the error
        log_can[0] = 1
        denominator_log = np.cumsum(np.log(log_can))
        for ll in range(self.LL):
            C_lower_V0 = cal_C_lower(np.intc(self.dataNum), np.intc(self.KK), self.connection.astype(np.intc), self.C_val[ll].astype(np.intc))
            D_lower_V0 = np.sum(self.D_val[ll], axis=1).astype(np.intc)  # sum k'' in D(ll)_i'k''k    ii=i'

            C_higher_V0 = np.sum(C_lower_V0, axis=1).astype(np.intc)
            D_higher_V0 = np.sum(D_lower_V0, axis=1).astype(np.intc)

            if ll==0:
                C_lower_V1 = cal_C_lower_together(np.intc(self.dataNum), np.intc(self.KK),
                                                    self.connection.astype(np.intc), self.C_val[ll].astype(np.intc), self.C_val[ll].astype(np.intc), ll)
                D_lower_V1 = np.sum(self.D_val[ll],axis=2).astype(np.intc)  # sum k'' in D(ll)_i k k''   ii=i
            else:
                C_lower_V1 = cal_C_lower_together(np.intc(self.dataNum), np.intc(self.KK),
                                                    self.connection.astype(np.intc), self.C_val[ll-1].astype(np.intc), self.C_val[ll].astype(np.intc), ll)

                D_lower_V1 = np.sum(self.D_val[ll - 1], axis=1).astype(np.intc)  # sum k'' in D(ll-1)_i k''k    ii=i
                D_lower_V1 = D_lower_V1+ np.sum(self.D_val[ll], axis=2).astype(np.intc)  # sum k'' in D(ll)_i k k''   ii=i

            C_higher_V1 = np.sum(C_lower_V1, axis=1).astype(np.intc)
            D_higher_V1 = np.sum(D_lower_V1, axis=1).astype(np.intc)

            if ll < (self.LL-1):
                C_lower_V2 = cal_C_lower_0(np.intc(self.dataNum), np.intc(self.KK),
                                                    self.connection.astype(np.intc), self.C_val[ll+1].astype(np.intc))
                D_lower_V2 = np.sum(self.D_val[ll + 1], axis=2).astype(np.intc)  # sum k'' in D(ll+1)_i'k k''  ii=i'
            elif ll == (self.LL-1):
                C_lower_V2 = copy.deepcopy(self.X_i).astype(np.intc)  # dataNum is for i' in ll+1
                D_lower_V2 = np.zeros((self.dataNum, self.KK)).astype(np.intc)   # sum k'' in D(ll+1)_i'k k''  ii=i'

            C_higher_V2 = np.sum(C_lower_V2, axis=1).astype(np.intc)
            D_higher_V2 = np.sum(D_lower_V2, axis=1).astype(np.intc)

            [self.C_val[ll], C_lower_V0, C_higher_V0, C_lower_V1, C_higher_V1] = update_C_integer(np.intc(bounded_version),alpha_k, alpha_k_sum, np.intc(self.KK),np.intc(len(candidates)),\
                             self.W_val[ll],self.M_c[ll], \
                             candidates.astype(np.intc),denominator_log, C_lower_V1, D_lower_V1, C_higher_V1,D_higher_V1,C_lower_V2, D_lower_V2, \
                             C_higher_V2,D_higher_V2, self.C_val[ll].astype(np.intc),C_lower_V0, C_higher_V0,self.connection.astype(np.intc), D_lower_V0, D_higher_V0)

            # self.D_val[ll] = update_D_integer(np.intc(bounded_version),alpha_k, alpha_k_sum, \
            #                  np.intc(self.dataNum), np.intc(self.KK),np.intc(len(candidates)),self.Psi_val,self.M[ll], \
            #                  candidates.astype(np.intc),denominator_log, C_lower_V1, D_lower_V1, C_higher_V1,D_higher_V1,C_lower_V2, D_lower_V2, \
            #                  C_higher_V2,D_higher_V2, self.D_val[ll].astype(np.intc),C_lower_V0, C_higher_V0,self.connection.astype(np.intc), D_lower_V0, D_higher_V0)

            # # update C
            # for ij in range(len(self.connection)):  # ii'
            #     for kk in range(self.KK):  # k
            #         conn_0 = self.connection[ij, 0] #i
            #         conn_1 = self.connection[ij, 1] #i'
            #         # part 1
            #         w_connection_ii = self.W_val[conn_0, conn_1]  # wii'
            #         log_M_star = np.log(self.M[ll]) + np.log(w_connection_ii)
            #         # part 2_v0
            #         C_lower_V0[conn_1, kk] -= self.C_val[ll, ij, kk]
            #         C_higher_V0[conn_1] -= self.C_val[ll, ij, kk]
            #         V0_lower = 1 + C_lower_V0[conn_1, kk] + D_lower_V0[conn_1, kk]
            #         V0_higher = self.KK + C_higher_V0[conn_1] + D_higher_V0[conn_1]
            #         # part 3_v1
            #         C_lower_V1[conn_0, kk]  -= self.C_val[ll, ij, kk]
            #         C_higher_V1[conn_0] -= self.C_val[ll, ij, kk]
            #         V1_lower = 1 + C_lower_V1[conn_0, kk]  + D_lower_V1[conn_0, kk]
            #         V1_higher = self.KK + C_higher_V1[conn_0] + D_higher_V1[conn_0]
            #         # part 4_v2
            #         V2_lower = V0_lower + C_lower_V2[conn_1, kk] + D_lower_V2[conn_1, kk]
            #         V2_higher = V0_higher + C_higher_V2[conn_1] + D_higher_V2[conn_1]
            #
            #         ######### use gammaln
            #         ll_posterior = candidates * log_M_star - denominator_log + gammaln(
            #             V0_higher + candidates) - gammaln(V0_lower + candidates) + gammaln(V1_lower+candidates) - gammaln(V1_higher+candidates) + gammaln(V2_lower+candidates) - gammaln(V2_higher+candidates)
            #
            #         probability_ii = np.exp(ll_posterior - np.max(ll_posterior))
            #         self.C_val[ll, ij, kk] = np.random.choice(candidates, p=probability_ii / np.sum(probability_ii))
            #         C_lower_V0[conn_1, kk] += self.C_val[ll, ij, kk]
            #         C_higher_V0[conn_1] += self.C_val[ll, ij, kk]
            #         C_lower_V1[conn_0, kk] += self.C_val[ll, ij, kk]
            #         C_higher_V1[conn_0] += self.C_val[ll, ij, kk]
            #
            #
            for ii in range(self.dataNum):  # i
                for kk1 in range(self.KK): # k
                    for kk2 in range(self.KK): # k'
                        # part 1
                        Psi_connection_ii = self.Psi_val[ll, kk1, kk2] # Psi_kk'
                        log_M_star = np.log(self.M_d[ll]) + np.log(Psi_connection_ii)
                        # part 2_v0
                        D_lower_V0[ii, kk2] -= self.D_val[ll, ii, kk1, kk2]
                        D_higher_V0[ii] -= self.D_val[ll, ii, kk1, kk2]
                        D_lower_V1[ii, kk1] -= self.D_val[ll, ii, kk1, kk2]
                        D_higher_V1[ii] -= self.D_val[ll, ii, kk1, kk2]

                        V0_lower = 1 + D_lower_V0[ii, kk2] + C_lower_V0[ii, kk2]
                        V0_higher = self.KK + D_higher_V0[ii] + C_higher_V0[ii]
                        # part 3_v1
                        V1_lower = 1 + C_lower_V1[ii, kk1] + D_lower_V1[ii, kk1]
                        V1_higher = self.KK + C_higher_V1[ii] + D_higher_V1[ii]

                        # part 4_v2
                        V2_lower = V0_lower + C_lower_V2[ii, kk2] + D_lower_V2[ii, kk2]
                        V2_higher = V0_higher + C_higher_V2[ii] + D_higher_V2[ii]

                        ######### use gammaln
                        ll_posterior = candidates * log_M_star - denominator_log + gammaln(
                            V0_higher + candidates) - gammaln(V0_lower + candidates) + gammaln(
                            V1_lower + candidates) - gammaln(V1_higher + candidates) + gammaln(
                            V2_lower + candidates) - gammaln(V2_higher + candidates)

                        probability_ii = np.exp(ll_posterior - np.max(ll_posterior))
                        self.D_val[ll, ii, kk1, kk2] = np.random.choice(candidates, p=probability_ii / np.sum(probability_ii))
                        D_lower_V0[ii, kk2] += self.D_val[ll, ii, kk1, kk2]
                        D_higher_V0[ii] += self.D_val[ll, ii, kk1, kk2]
                        D_lower_V1[ii, kk1] += self.D_val[ll, ii, kk1, kk2]
                        D_higher_V1[ii] += self.D_val[ll, ii, kk1, kk2]


    def sample_X_i_integer(self, dataR_matrix):

        # start_time = time.time()
        # for i in range(10):
        #     random_val = uniform.rvs(size=(1000, 1000))
        #     b = np.sum(random_val)
        # print('vec sum time: ', time.time()-start_time)
        #
        #
        # start_time = time.time()
        # for i in range(10):
        #     random_val = uniform.rvs(size=(1000, 1000))
        #     b = simple_sum(random_val, 1000)
        # print('cython sum time: ', time.time()-start_time)


        # sample the latent counts X_i

        idx = (dataR_matrix != (-1))
        np.fill_diagonal(idx, False) # not need to consider i==j

        candidates_1 = np.arange(1, 1000+1)
        candidates_1_log = np.log(candidates_1)

        candidates_0 = np.arange(0, 1000+1)
        candidates_0_log = np.log(np.arange(1, 1000+1))
        candidates_0_log = np.concatenate(([0], candidates_0_log))

        # for C
        ##################################  V0  ##########################################
        C_lower_V0 = np.zeros((self.dataNum, self.KK))  # dataNum is for i' in L
        for ij1 in range(len(self.connection)):
            C_lower_V0[self.connection[ij1, 1]] += self.C_val[-1, ij1]  # sum i'' in Ci''k i'(ll)
        C_higher_V0 = np.sum(C_lower_V0, axis=1)
        ## for D
        ##################################  V0  ##############################################
        D_lower_V0 = np.sum(self.D_val[-1], axis=1)  # sum k'' in D(ll)_i'k''k    ii=i'
        D_higher_V0 = np.sum(D_lower_V0, axis=1)

        for nn in range(self.dataNum):
            Xik_Lambda = np.sum(np.dot(self.Lambdas, (idx[nn][:, np.newaxis] * self.X_i).T), axis=1) + \
                         np.sum(np.dot(self.Lambdas.T, (idx[:, nn][:, np.newaxis] * self.X_i).T), axis=1)
            log_alpha_X = np.log(self.M_X)  - Xik_Lambda
            for kk in range(self.KK):
                n_X = self.Z_ik[nn, kk]
                V0_lower = 1 + D_lower_V0[nn, kk] + C_lower_V0[nn, kk]
                V0_higher = self.KK + D_higher_V0[nn] + C_higher_V0[nn]
                if n_X == 0:
                    pseudos = candidates_0 * log_alpha_X[kk] - np.cumsum(
                        candidates_0_log) + gammaln(V0_lower+candidates_0) -gammaln(V0_higher+np.sum(self.X_i[nn])-self.X_i[nn, kk]+candidates_0)
                    proportions = np.exp(pseudos - max(pseudos))
                    select_val = np.random.choice(candidates_0, p=proportions/np.sum(proportions))
                else:
                    pseudos = candidates_1 * log_alpha_X[kk] + n_X * candidates_1_log - np.cumsum(
                        candidates_1_log) + gammaln(V0_lower+candidates_1) -gammaln(V0_higher+np.sum(self.X_i[nn])-self.X_i[nn, kk]+candidates_1)
                    proportions = np.exp(pseudos - max(pseudos))
                    select_val = np.random.choice(candidates_1, p=proportions/np.sum(proportions))
                self.X_i[nn, kk] = select_val
