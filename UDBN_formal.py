import numpy as np
import scipy
import copy
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import timeit
import time
import csv
import os
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import poisson, norm, gamma, bernoulli
import scipy.io as sio

from utility_update import *


if __name__ == '__main__':
    # pathss_save = ''
    # pathss_save = '/scratch/xc35/xf5259/Inter_DBN/'
    pathss_save = 'result/'
    ##########################Dec_Data########################
    # pathss = '/Users/13106240/AppLockerExceptions/PycharmProjects/wider_and_deeper_DBN/UDBN/data/Dec_data/'
    # pathss = '/scratch/xc35/xf5259/data/'
    # pathss = '../data/'
    # filess = glob.glob(pathss+'*relation*.mat')
    pathss = '../Code/data/'
    # filess = glob.glob(pathss + '*AstroPh*.mat')
    # pathss = ''
    filess = glob.glob(pathss + 'CA-AstroPh*.mat')


    filess_i = filess[0]
    name_i = filess_i[len(pathss):(-4)]
    # print(name_i)
    #### read relational data
    # Training data(N*N, test position = -1), N, test data(N*N, training position = -1)
    dataR_matrix, dataNum, test_relation = load_data_fan(filess_i)

    # dataR: L_training x 2 format(Rij=1 NO Rii). each row of dataR records a relationship[i,j], dataR[ij][0] relates to dataR[ij][1]
    dataR = np.asarray(np.where(dataR_matrix==1)).T

    # len(np.unique(dataR))
    # np.max(dataR)
    # dataR_test: L_test x 2 format(Location of test data YES Rii, but removed in the next step
    dataR_test = np.asarray(np.where(test_relation!=-1)).T
    # dataR_test_val: extract test data as a array, YES Rii, but removed in the next step
    dataR_test_val = test_relation[test_relation!=-1]
    ## This is to remove the self-connected links in dataR_test(location)& dataR_test_val(relation array)
    notdelete_index = (dataR_test[:, 0]!=dataR_test[:, 1]) # index of Rij,i!=j
    dataR_test = dataR_test[notdelete_index] # dataR_test: Location NO Rii
    dataR_test_val = dataR_test_val[notdelete_index] # dataR_test_val: array NO Rii


    case_number = 1 # 1: only C; 2: only D; 3: both C and D.
    bounded_version = 0 # 1: yes, unbounded; 0: no, fixed number of communities.
    Integer_version = 0 # 1: yes, using deep integer networks; 0: no, keep on the \pi variables
    KK = 20 # length of \pi in the layer
    LL = 3 # number of layers = LL+1
    IterationTime = 2000

    # print('DataNum: ', dataNum)
    # print('LinkNum: ', len(dataR))
    current_time = time.time()
    M, X_i, Lambdas, Z_ik, Z_k1k2, pis, C_val, D_val, W_val, Psi_val, connections = initialize_model(dataR, dataNum, KK, LL, case_number, bounded_version)
    elapse_time = time.time()-current_time
    print('Elapsed time is: ', elapse_time)
    UDBN = UDBN_class(connections, dataNum, LL, KK, M, X_i, Lambdas, Z_ik, Z_k1k2, pis, C_val, D_val, W_val, Psi_val)

    # test_scores = UDBN.predictTestScore(test_relation)
    # test_data = test_relation[test_relation!=(-1)]
    # auc_seq[0] = roc_auc_score(test_data, test_scores)
    # test_ll_seq[0] = np.sum(np.log(test_scores)*(test_data==1))+np.sum(np.log(1-test_scores)*(test_data==0))


    #### model sampling and evaluation
    test_precision_seq = []
    auc_seq = []

    mean_predict = 0

    mean_pis = 0
    mean_W_val = [0 for _ in range(LL)]
    mean_Phi_val = [0 for _ in range(LL)]
    mean_xi = 0
    mean_C_val = 0
    mean_D_val = 0
    mean_lambda = 0
    mean_r_k = 0
    mean_epsilons = 0

    ids = uniform.rvs()
    collectionTime = int(IterationTime/2)
    # collectionTime = 30

    each_auc_seq = []

    start_time = time.time()
    for ite in range(IterationTime):

        # start_time = time.time()
        UDBN.sample_Z_ik_k1k2(dataR)
        if Integer_version == 0:
            UDBN.sample_X_i(dataR_matrix)
            UDBN.sample_C_D_Pi(case_number, bounded_version)
        elif Integer_version == 1:
            UDBN.sample_C_D_integer_process(bounded_version)
            UDBN.sample_X_i_integer(dataR_matrix)

        UDBN.sample_w_psi(case_number)

        UDBN.sample_M()
        UDBN.sample_Lambda_k1k2(dataR_matrix)
        # print(UDBN.alpha_omega)
        # print(UDBN.M)
        # elapse_time_numba_1 = time.time()-start_time
        if np.mod(ite, 10)==0:
            print('Iteration finished: ', ite)
            print('Per iteration running time: ', time.time()-start_time)
            start_time = time.time()
        # print(UDBN.xi_omega)
        # print(ite)
        predicted_val_inte = np.sum((UDBN.X_i[dataR_test[:, 0]][:, :, np.newaxis] * UDBN.X_i[dataR_test[:, 1]][:,np.newaxis, :]) * UDBN.Lambdas[np.newaxis, :, :],axis=(1, 2))
        predicted_val = 1 - np.exp(-predicted_val_inte)
        each_auc_seq.append(roc_auc_score(dataR_test_val, predicted_val))
        # print('Current AUC: ', each_auc_seq[-1])

        if (ite >collectionTime):

            mean_predict = (mean_predict*(ite - collectionTime-1)+predicted_val)/(ite-collectionTime)

            mean_pis = (mean_pis*(ite - collectionTime-1)+UDBN.pis)/(ite-collectionTime)
            mean_W_val = [(mean_W_val[ll]*(ite - collectionTime-1)+UDBN.W_val[ll])/(ite-collectionTime) for ll in range(LL)]
            mean_Phi_val = [(mean_Phi_val[ll]*(ite - collectionTime-1)+UDBN.Psi_val[ll])/(ite-collectionTime) for ll in range(LL)]
            mean_xi = (mean_xi*(ite - collectionTime-1)+UDBN.X_i)/(ite-collectionTime)
            mean_C_val = (mean_C_val*(ite - collectionTime-1)+UDBN.C_val)/(ite-collectionTime)
            mean_D_val = (mean_D_val*(ite - collectionTime-1)+UDBN.D_val)/(ite-collectionTime)
            mean_lambda = (mean_lambda*(ite - collectionTime-1)+UDBN.Lambdas)/(ite-collectionTime)

            mean_r_k = (mean_r_k*(ite - collectionTime-1)+UDBN.r_k)/(ite-collectionTime)
            mean_epsilons = (mean_epsilons*(ite - collectionTime-1)+UDBN.epsilon)/(ite-collectionTime)

            current_AUC = roc_auc_score(dataR_test_val, mean_predict)
            # print(current_AUC)
            current_precision = average_precision_score(dataR_test_val, mean_predict)
            auc_seq.append(current_AUC)
            test_precision_seq.append(current_precision)

            if np.mod(ite, 20)==0:
                # np.savez_compressed(
                #     pathss_save + 'UDBN_' + name_i + '_LL_' + str(LL) + '_KK_' + str(KK) + '_' + str(ids), auc_seq=auc_seq)

                plt.plot(auc_seq)
                plt.savefig(pathss_save+name_i + '_casenum_'+str(case_number)+'_K_'+str(KK)+'_L_'+str(LL)+'_id_' + str(ids) + '.pdf', edgecolor='none', format='pdf',
                            bbox_inches='tight')
                plt.close()

                np.savez_compressed(pathss_save + name_i + '_casenum_'+str(case_number)+'_LL_' + str(LL) + '_KK_' + str(KK) + '_' + str(ids), pis = mean_pis, mean_W_val = mean_W_val, mean_Phi_val = mean_Phi_val,
                    C_val = mean_C_val, D_val = mean_D_val, lambdas = mean_lambda, xi = mean_xi, test_precision_seq=test_precision_seq, auc_seq=auc_seq, dataNum = dataNum,
                    LL = LL, KK = KK, mean_r_k = mean_r_k, mean_epsilons = mean_epsilons, case = case_number, bounded_version= bounded_version, each_auc_seq = each_auc_seq)

