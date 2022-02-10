# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3, profile=False
import numpy as np
import time
from scipy.special import gammaln
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp

def simple_sum(double[:, :] a_val, int Len):
    cdef:
        int i
        double[:] re_val = np.zeros(Len)
    for i in range(Len):
        for j in range(Len):
            re_val[i] += a_val[i,j]
    return re_val

def CRT_sample(int[:, :] Z_k1k2, double[:, :] R_KK):
    cdef:
        int k1, k2, k3, Len_0 = len(Z_k1k2), Len_1 = len(Z_k1k2[0])
        int[:,:] L_KK = np.zeros((Len_0, Len_1), dtype=np.intc)
        double rr

    for k1 in range(Len_0):
        for k2 in range(Len_1):
            if Z_k1k2[k1][k2]>0:
                for k3 in range(Z_k1k2[k1][k2]):
                    rr = <double>rand() / <double>RAND_MAX
                    if rr<(R_KK[k1][k2]/(R_KK[k1][k2]+k3)):
                        L_KK[k1][k2] += 1
    return L_KK


def CRT_sample_vector(int[:] Z_k1k2, double[:] R_KK):
    cdef:
        int k1, k3, Len_0 = len(Z_k1k2)
        int[:] L_KK = np.zeros(Len_0, dtype=np.intc)
        double rr

    for k1 in range(Len_0):
        if Z_k1k2[k1]>0:
            for k3 in range(Z_k1k2[k1]):
                rr = <double>rand() / <double>RAND_MAX
                if rr<(R_KK[k1]/(R_KK[k1]+k3)):
                    L_KK[k1] += 1
    return L_KK


def CRT_sample_binomial(int[:, :] Z_k1k2, double[:, :] alpha_KK, double[:, :] NN_KK):
    cdef:
        int k1, k2, k3, Len_0 = len(Z_k1k2), Len_1 = len(Z_k1k2[0])
        int[:,:] L_KK = np.zeros((Len_0, Len_1), dtype=np.intc)
        double rr

    for k1 in range(Len_0):
        for k2 in range(Len_1):
            if Z_k1k2[k1][k2]>0:
                for k3 in range(Z_k1k2[k1][k2]):
                    rr = <double>rand() / <double>RAND_MAX
                    if rr<(alpha_KK[k1][k2]/(alpha_KK[k1][k2]+k3+NN_KK[k1][k2])):
                        L_KK[k1][k2] += 1
    return L_KK


cdef int positive_poisson_sample_cython(double z_lambda):
    # return positive truncated poisson random variables Z = 1, 2, 3, 4, ...
    # z_lambda: parameter for Poisson distribution

    cdef:
        int k = 1
        double log_r
        double s = 0.
        double log_current_val = 0.
        double prob_sum
        double log_z_lambda = log(z_lambda)

    prob_sum = 1-exp(-z_lambda)

    # Generate random number
    log_r = log(<double>rand() / <double>RAND_MAX)+ log(prob_sum)+z_lambda

    # Generate topic

    while(True):
        log_current_val = log_current_val + log_z_lambda-log(k)
        s += exp(log_current_val)
        if log_r < log(s):
            break
        k += 1

        if k>1000:
            break

    return k


def positive_poisson_sample_cyuse(double z_lambda):
    # return positive truncated poisson random variables Z = 1, 2, 3, 4, ...
    # z_lambda: parameter for Poisson distribution

    cdef:
        int k = 1
        double log_r
        double s = 0.
        double log_current_val = 0.
        double prob_sum
        double log_z_lambda = log(z_lambda)

    prob_sum = 1-exp(-z_lambda)

    # Generate random number
    log_r = log(<double>rand() / <double>RAND_MAX)+ log(prob_sum)+z_lambda

    # Generate topic

    while(True):
        log_current_val = log_current_val + log_z_lambda-log(k)
        s += exp(log_current_val)
        if log_r < log(s):
            break
        k += 1

        if k>1000:
            break

    return k


def sample_z_cython(int reNum, int KK, int dataNum, int[:, :] dataR, double[:, :] X_i, double[:, :] Lambdas):
    cdef:
        Py_ssize_t ii, kk1, kk2
        double[:, :] pois_lambda = np.zeros((KK, KK))
        int[:, :] Z_k1k2 = np.zeros((KK, KK), dtype=np.intc)
        int[:, :] Z_ik = np.zeros((dataNum, KK), dtype=np.intc)
        double pois_lambda_sum, r, current_val
        int total_val

    for ii in range(reNum):
        pois_lambda_sum = 0.
        for kk1 in range(KK):
            for kk2 in range(KK):
                pois_lambda[kk1][kk2] = X_i[dataR[ii][0]][kk1]*Lambdas[kk1][kk2]*X_i[dataR[ii][1]][kk2]
                pois_lambda_sum += pois_lambda[kk1][kk2]

        total_val = positive_poisson_sample_cython(pois_lambda_sum)

        for v in range(total_val):
            r = <double>rand() / <double>RAND_MAX*pois_lambda_sum
            current_val = 0.
            for kk1 in range(KK):
                for kk2 in range(KK):
                    current_val += pois_lambda[kk1][kk2]
                    if r<current_val:
                        Z_k1k2[kk1][kk2] += 1
                        Z_ik[dataR[ii][0]][kk1] += 1
                        Z_ik[dataR[ii][1]][kk2] += 1
                        break
                if r<current_val:
                    break

    return Z_k1k2, Z_ik

def cal_C_lower(int dataNum, int KK, int[:, ::1] connection, int[:, ::1] C_val):
    cdef:
        int[:, :] C_lower_V0 = np.zeros((dataNum, KK), dtype=np.intc)
        Py_ssize_t ij1, kk
    for ij1 in range(len(connection)):
        for kk in range(KK):
            C_lower_V0[connection[ij1][1]][kk] += C_val[ij1][kk]
    return C_lower_V0

def cal_C_lower_0(int dataNum, int KK, int[:, ::1] connection, int[:, ::1] C_val):
    cdef:
        int[:, :] C_lower_V0_0 = np.zeros((dataNum, KK), dtype=np.intc)
        Py_ssize_t ij1, kk
    for ij1 in range(len(connection)):
        for kk in range(KK):
            C_lower_V0_0[connection[ij1][0]][kk] += C_val[ij1][kk]
    return C_lower_V0_0

def cal_C_lower_together(int dataNum, int KK, int[:, ::1] connection, int[:, ::1] C_val_minus, int[:, ::1] C_val, int ll):
    cdef:
        int[:, :] C_lower_V0_V1 = np.zeros((dataNum, KK), dtype=np.intc)
        Py_ssize_t ij1, kk
    for ij1 in range(len(connection)):
        for kk in range(KK):
            C_lower_V0_V1[connection[ij1][0]][kk] += C_val[ij1][kk]
            if ll>0:
                C_lower_V0_V1[connection[ij1][1]][kk] += C_val_minus[ij1][kk]

    return C_lower_V0_V1


def update_C(int bounded_version, double[:] alpha_k, int KK, Py_ssize_t can_len, double[:, ::1] pis_ll, \
             double[:, ::1] pis_ll_1, double[:, ::1] W_val, double M, int [::1] candidates, double[::1] denominator_log,
             int[:, :] C_val, int[:, ::1] C_lower_V0, int[:] C_higher_V0, int[:, ::1] connection, int[:, ::1] D_lower_V0, int[:] D_higher_V0):
    # update C
    cdef:
        double pi_lik_from_i, pi_lik_to_i, Psi_connection_ii, log_M_star, v_lower, v_higher, max_ll, prob_sum, gammaln_val
        double[::1] ll_posterior, pro_seq
        int k0, conn_0, conn_1
        Py_ssize_t ij, kk, kk3

    ll_posterior = np.zeros(can_len)
    pro_seq = np.zeros(can_len)

    for ij in range(len(connection)): # ii'``
        for kk in range(KK): # k
            ###################### This is the finite states version
            conn_0 = connection[ij][0]
            conn_1 = connection[ij][1]
            # part 1
            pi_lik_from_i = pis_ll[conn_0][kk] # pis(l)_ik
            pi_lik_to_i = pis_ll_1[conn_1][kk]# pis(l+1)_i'k
            Psi_connection_ii = W_val[conn_0][conn_1] # wii'
            log_M_star = log(M) + log(pi_lik_from_i) + log(Psi_connection_ii) + log(pi_lik_to_i)
            # part 2
            C_lower_V0[conn_1][kk] -= C_val[ij][kk]
            C_higher_V0[conn_1] -= C_val[ij][kk]

            v_lower = alpha_k[kk] + D_lower_V0[conn_1][kk] + C_lower_V0[conn_1][kk]
            v_higher = sum(alpha_k) + D_higher_V0[conn_1] + C_higher_V0[conn_1] # here, not -1, np.arrange will -1


            gammaln_val = gammaln(v_higher + candidates[0]) - gammaln(v_lower + candidates[0])
            # gammaln_val = 0
            for kk3 in range(can_len):
                if kk3 >0:
                    gammaln_val = gammaln_val - log((v_lower+candidates[kk3-1])/(v_higher+candidates[kk3-1]))
                ll_posterior[kk3] = candidates[kk3] * log_M_star - denominator_log[kk3] + gammaln_val
                if kk3 ==0:
                    max_ll = ll_posterior[kk3]
                else:
                    max_ll = max(max_ll, ll_posterior[kk3])

            prob_sum = 0
            for kk3 in range(can_len):
                ll_posterior[kk3] -= max_ll
                pro_seq[kk3] = exp(ll_posterior[kk3])
                prob_sum += pro_seq[kk3]
            for kk3 in range(can_len):
                pro_seq[kk3] = pro_seq[kk3]/prob_sum

            k0 = categoricals(can_len, pro_seq)

            C_val[ij][kk] = k0
            C_lower_V0[conn_1][kk] += C_val[ij][kk]
            C_higher_V0[conn_1] += C_val[ij][kk]
    return C_val

def update_D(int bounded_version, double[:] alpha_k, int dataNum, int KK, Py_ssize_t can_len, double[:, ::1] pis_ll, double[:, ::1] pis_ll_1, double[:, ::1] Psi_val, double M, int [::1] candidates, double[::1] denominator_log,
             int[:, :, ::1] D_val, int[:, ::1] D_lower_V0, int[::1] D_higher_V0, int[:, ::1] C_lower_V0, int[::1] C_higher_V0):

    cdef:
        double pi_lik_from_i, pi_lik_to_i, Psi_connection_ii, log_M_star, v_lower, v_higher, max_ll, prob_sum, gammaln_val
        double[::1] ll_posterior, pro_seq
        # int[:, ::1] D_lower_V0
        # int[::1] D_higher_V0
        # DD = np.zeros((dataNum, KK, KK), dtype = np.intc)
        # int[:, :,:] D_val = DD
        int k0
        Py_ssize_t ii, kk1, kk2, kk3

    # D_lower_V0 = D_lower
    # D_higher_V0 = D_higher
    #
    ll_posterior = np.zeros(can_len)
    pro_seq = np.zeros(can_len)
    #
    for ii in range(dataNum): # i
        for kk1 in range(KK): # k
            for kk2 in range(KK): # k'

                # part 1
                pi_lik_from_i = pis_ll[ii][kk1] # pis(l)_ik
                pi_lik_to_i = pis_ll_1[ii][kk2] # pis(l+1)_ik'
                Psi_connection_ii = Psi_val[kk1][kk2] # Psi_kk'
                log_M_star = log(M) + log(pi_lik_from_i) + log(Psi_connection_ii) + log(pi_lik_to_i)
                # part 2
                D_lower_V0[ii, kk2] -= D_val[ii][kk1][kk2]
                D_higher_V0[ii] -= D_val[ii][kk1][kk2]


                v_lower = alpha_k[kk2] + D_lower_V0[ii][kk2]+ C_lower_V0[ii,kk2]
                v_higher = sum(alpha_k) + D_higher_V0[ii] + C_higher_V0[ii]  # here, not -1, np.arrange will -1

                ######### use gammaln
                gammaln_val = gammaln(v_higher + candidates[0]) - gammaln(v_lower + candidates[0])
                # gammaln_val = 0

                for kk3 in range(can_len):
                    if kk3 >0:
                        gammaln_val = gammaln_val - log((v_lower+candidates[kk3-1])/(v_higher+candidates[kk3-1]))
                    ll_posterior[kk3] = candidates[kk3] * log_M_star - denominator_log[kk3]+gammaln_val
                    if kk3 ==0:
                        max_ll = ll_posterior[kk3]
                    else:
                        max_ll = max(max_ll, ll_posterior[kk3])

                prob_sum = 0
                for kk3 in range(can_len):
                    ll_posterior[kk3] -= max_ll
                    pro_seq[kk3] = exp(ll_posterior[kk3])
                    prob_sum += pro_seq[kk3]
                for kk3 in range(can_len):
                    pro_seq[kk3] = pro_seq[kk3]/prob_sum

                k0 = categoricals(can_len, pro_seq)
                D_val[ii, kk1, kk2] = k0

                D_lower_V0[ii,kk2] += D_val[ii, kk1, kk2]
                D_higher_V0[ii] += D_val[ii, kk1, kk2]

    return D_val



def update_C_integer(int bounded_version, double[:] alpha_k, double alpha_k_sum, int KK, Py_ssize_t can_len, double[:, ::1] W_val, double M, \
                     int [::1] candidates, double[::1] denominator_log, int[:, :] C_lower_V1, int[:, :] D_lower_V1, int[:] C_higher_V1, int[:] D_higher_V1, \
                     int[:, :] C_lower_V2, int[:,:] D_lower_V2, int[:] C_higher_V2, int[:] D_higher_V2, \
                    int[:, :] C_val, int[:, ::1] C_lower_V0, int[:] C_higher_V0, int[:, ::1] connection, int[:, ::1] D_lower_V0, int[:] D_higher_V0):
    # update C
    cdef:
        double Psi_connection_ii, log_M_star, v_lower_v0, v_higher_v0, v_lower_v1, v_higher_v1, v_lower_v2, v_higher_v2, max_ll, prob_sum, gammaln_val
        double[::1] ll_posterior, pro_seq
        int k0, conn_0, conn_1
        Py_ssize_t ij, kk, kk3

    ll_posterior = np.zeros(can_len)
    pro_seq = np.zeros(can_len)

    for ij in range(len(connection)): # ii'``
        for kk in range(KK): # k
            ###################### This is the finite states version
            conn_0 = connection[ij][0]
            conn_1 = connection[ij][1]
            # part 1
            Psi_connection_ii = W_val[conn_0][conn_1] # wii'
            log_M_star = log(M) + log(Psi_connection_ii)
            # part 2
            C_lower_V0[conn_1][kk] -= C_val[ij][kk]
            C_higher_V0[conn_1] -= C_val[ij][kk]
            C_lower_V1[conn_0, kk]  -= C_val[ij][kk]
            C_higher_V1[conn_0] -= C_val[ij][kk]

            v_lower_v0 = alpha_k[kk] + D_lower_V0[conn_1][kk] + C_lower_V0[conn_1][kk]
            v_higher_v0 = alpha_k_sum + D_higher_V0[conn_1] + C_higher_V0[conn_1] # here, not -1, np.arrange will -1
            gammaln_val = gammaln(v_higher_v0 + candidates[0]) - gammaln(v_lower_v0 + candidates[0])

            v_lower_v1 = alpha_k[kk] + C_lower_V1[conn_0, kk]  + D_lower_V1[conn_0, kk]
            v_higher_v1 = alpha_k_sum + C_higher_V1[conn_0] + D_higher_V1[conn_0]
            gammaln_val += (gammaln(v_lower_v1 + candidates[0]) -gammaln(v_higher_v1 + candidates[0]))

            v_lower_v2 = v_lower_v0 + C_lower_V2[conn_1, kk] + D_lower_V2[conn_1, kk]
            v_higher_v2 = v_higher_v0 + C_higher_V2[conn_1] + D_higher_V2[conn_1]
            gammaln_val += (gammaln(v_lower_v2 + candidates[0]) - gammaln(v_higher_v2 + candidates[0]))

            for kk3 in range(can_len):
                if kk3 >0:
                    gammaln_val = gammaln_val - log((v_lower_v0+candidates[kk3-1])*(v_higher_v1+candidates[kk3-1])*(v_higher_v2+candidates[kk3-1])\
                                                    /((v_higher_v0+candidates[kk3-1])*(v_lower_v1+candidates[kk3-1])*(v_lower_v2+candidates[kk3-1])))
                ll_posterior[kk3] = candidates[kk3] * log_M_star - denominator_log[kk3] + gammaln_val
                if kk3 ==0:
                    max_ll = ll_posterior[kk3]
                else:
                    max_ll = max(max_ll, ll_posterior[kk3])

            prob_sum = 0
            for kk3 in range(can_len):
                ll_posterior[kk3] -= max_ll
                pro_seq[kk3] = exp(ll_posterior[kk3])
                prob_sum += pro_seq[kk3]
            for kk3 in range(can_len):
                pro_seq[kk3] = pro_seq[kk3]/prob_sum

            k0 = categoricals(can_len, pro_seq)

            C_val[ij][kk] = k0
            C_lower_V0[conn_1][kk] += C_val[ij][kk]
            C_higher_V0[conn_1] += C_val[ij][kk]
            C_lower_V1[conn_0, kk] += C_val[ij, kk]
            C_higher_V1[conn_0] += C_val[ij, kk]

    return C_val, C_lower_V0, C_higher_V0, C_lower_V1, C_higher_V1



def update_D_integer(int bounded_version, double[:] alpha_k, double alpha_k_sum, int dataNum, int KK, Py_ssize_t can_len, double[:, ::1] Psi_val, double M, \
                     int [::1] candidates, double[::1] denominator_log, int[:, :] C_lower_V1, int[:, :] D_lower_V1, int[:] C_higher_V1, int[:] D_higher_V1, \
                     int[:, :] C_lower_V2, int[:,:] D_lower_V2, int[:] C_higher_V2, int[:] D_higher_V2, \
                    int[:, :, :] D_val, int[:, ::1] C_lower_V0, int[:] C_higher_V0, int[:, ::1] connection, int[:, ::1] D_lower_V0, int[:] D_higher_V0):
    cdef:
        double Psi_connection_ii, log_M_star, v_lower_v0, v_higher_v0, v_lower_v1, v_higher_v1, v_lower_v2, v_higher_v2, max_ll, prob_sum, gammaln_val
        double[::1] ll_posterior, pro_seq
        int k0
        Py_ssize_t ii, kk1, kk2, kk3

    ll_posterior = np.zeros(can_len)
    pro_seq = np.zeros(can_len)
    #
    for ii in range(dataNum): # i
        for kk1 in range(KK): # k
            for kk2 in range(KK): # k'

                # part 1
                Psi_connection_ii = Psi_val[kk1][kk2] # Psi_kk'
                log_M_star = log(M) + log(Psi_connection_ii)
                # part 2
                D_lower_V0[ii, kk2] -= D_val[ii][kk1][kk2]
                D_higher_V0[ii] -= D_val[ii][kk1][kk2]
                D_lower_V1[ii, kk1] -= D_val[ii, kk1, kk2]
                D_higher_V1[ii] -= D_val[ii, kk1, kk2]

                v_lower_v0 = alpha_k[kk2] + D_lower_V0[ii][kk2]+ C_lower_V0[ii,kk2]
                v_higher_v0 = alpha_k_sum + D_higher_V0[ii] + C_higher_V0[ii]  # here, not -1, np.arrange will -1
                gammaln_val = gammaln(v_higher_v0 + candidates[0]) - gammaln(v_lower_v0 + candidates[0])

                v_lower_v1 = alpha_k[kk2] + C_lower_V1[ii, kk1] + D_lower_V1[ii, kk1]
                v_higher_v1 = alpha_k_sum + C_higher_V1[ii] + D_higher_V1[ii]

                gammaln_val += (gammaln(v_lower_v1 + candidates[0]) - gammaln(v_higher_v1 + candidates[0]))

                # part 4_v2
                v_lower_v2 = v_lower_v0 + C_lower_V2[ii, kk2] + D_lower_V2[ii, kk2]
                v_higher_v2 = v_higher_v0 + C_higher_V2[ii] + D_higher_V2[ii]
                gammaln_val += (gammaln(v_lower_v2 + candidates[0]) - gammaln(v_higher_v2 + candidates[0]))


                for kk3 in range(can_len):
                    if kk3 >0:
                        gammaln_val = gammaln_val - log(((v_lower_v0+candidates[kk3-1])*(v_higher_v1+candidates[kk3-1])*(v_higher_v2+candidates[kk3-1]))\
                                                        /((v_higher_v0+candidates[kk3-1])*(v_lower_v1+candidates[kk3-1])*(v_lower_v2+candidates[kk3-1])))
                    ll_posterior[kk3] = candidates[kk3] * log_M_star - denominator_log[kk3]+gammaln_val
                    if kk3 ==0:
                        max_ll = ll_posterior[kk3]
                    else:
                        max_ll = max(max_ll, ll_posterior[kk3])

                prob_sum = 0
                for kk3 in range(can_len):
                    ll_posterior[kk3] -= max_ll
                    pro_seq[kk3] = exp(ll_posterior[kk3])
                    prob_sum += pro_seq[kk3]
                for kk3 in range(can_len):
                    pro_seq[kk3] = pro_seq[kk3]/prob_sum

                k0 = categoricals(can_len, pro_seq)

                D_val[ii, kk1, kk2] = k0
                D_lower_V0[ii,kk2] += D_val[ii, kk1, kk2]
                D_higher_V0[ii] += D_val[ii, kk1, kk2]
                D_lower_V1[ii, kk1] += D_val[ii, kk1, kk2]
                D_higher_V1[ii] += D_val[ii, kk1, kk2]

    return D_val




cdef int categoricals(int n, double[::1] p):  # k = 0,...,n-1
    cdef:
        int k = 0
        double r = 0.
        double s = 0.
    # Generate random number
    r = <double>rand() / <double>RAND_MAX
    # Generate topic
    for k in range(n):
        s += p[k]
        if r < s:
            break
    return k
