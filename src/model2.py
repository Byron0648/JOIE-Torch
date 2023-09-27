import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiG import multiG
import pickle
from utils import circular_correlation, np_ccorr


class joie(nn.Module):
    def __init__(self, num_rels1, num_ents1, num_rels2, num_ents2, method='distmult', bridge='CG', dim1=300, dim2=100,
                 batch_sizeK1=512, batch_sizeK2=512, batch_sizeA=256, L1=False):
        super(joie, self).__init__()
        self.num_relsA = num_rels1
        self.num_entsA = num_ents1
        self.num_relsB = num_rels2
        self.num_entsB = num_ents2
        self.method = method
        self.bridge = bridge
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hid_dim = 50
        self.batch_sizeK1 = batch_sizeK1
        self.batch_sizeK2 = batch_sizeK2
        self.batch_sizeA = batch_sizeA
        self.epoch_loss = 0
        # margins
        self.m1 = 0.5
        self.m2 = 1.0
        self.mA = 0.5
        self.L1 = L1
        print("Pytorch Part build up! Embedding method: [" + self.method + "]. Bridge method:[" + self.bridge + "]")
        print("Margin Paramter: [m1] " + str(self.m1) + " [m2] " + str(self.m2))
        # KG1
        self.ht1 = nn.Parameter(torch.empty(self.num_entsA, self.dim1))
        self.r1 = nn.Parameter(torch.empty(self.num_relsA, self.dim1))
        # KG2
        self.ht2 = nn.Parameter(torch.empty(self.num_entsB, self.dim2))
        self.r2 = nn.Parameter(torch.empty(self.num_relsB, self.dim2))

        # Affine map and init
        self.M = nn.Parameter(torch.empty(self.dim1, self.dim2))
        self.b = nn.Parameter(torch.empty(self.dim2))
        self.Mc = nn.Parameter(torch.empty(self.dim2, self.hidden_dim))
        self.bc = nn.Parameter(torch.empty(self.hidden_dim))
        self.Me = nn.Parameter(torch.empty(self.dim1, self.hidden_dim))
        self.be = nn.Parameter(torch.empty(self.hidden_dim))


        # initializer
        nn.init.uniform_(self.ht1)
        nn.init.uniform_(self.r1)
        nn.init.uniform_(self.ht2)
        nn.init.uniform_(self.r2)
        nn.init.orthogonal_(self.M)  # initializer=orthogonal_initializer()
        nn.init.normal_(self.b)  # initializer=tf.truncated_normal_initializer
        nn.init.orthogonal_(self.Mc)  # initializer=orthogonal_initializer(), dtype=tf.float32)
        nn.init.normal_(self.bc)  # initializer=tf.truncated_normal_initializer, dtype=tf.float32)
        nn.init.orthogonal_(self.Me)  # initializer=orthogonal_initializer(), dtype=tf.float32)
        nn.init.normal_(self.be)  # initializer=tf.truncated_normal_initializer, dtype=tf.float32)



    def forwardA(self, A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index):
        # KG1
        A_h_ent_batch = F.normalize(self.ht1[A_h_index], p=2, dim=1)
        A_t_ent_batch = F.normalize(self.ht1[A_t_index], p=2, dim=1)
        A_rel_batch = self.r1[A_r_index]
        A_hn_ent_batch = F.normalize(self.ht1[A_hn_index], p=2, dim=1)
        A_tn_ent_batch = F.normalize(self.ht1[A_tn_index], p=2, dim=1)

        if self.method == 'transe':
            ##### TransE score
            # This stores h + r - t
            A_loss_matrix = A_h_ent_batch + A_rel_batch - A_t_ent_batch
            # This stores h' + r - t' for negative samples
            A_neg_matrix = A_hn_ent_batch + A_rel_batch - A_tn_ent_batch
            if self.L1:
                A_loss = torch.sum(
                    torch.max(torch.sum(torch.abs(A_loss_matrix), dim=1) + self.m1
                              - torch.sum(torch.abs(A_neg_matrix), dim=1), torch.tensor(0.))
                ) / self.batch_sizeK1
            else:
                A_loss = torch.sum(
                    torch.max(torch.sqrt(torch.sum(torch.square(A_loss_matrix), dim=1)) + self.m1
                              - torch.sqrt(torch.sum(torch.square(A_neg_matrix), dim=1)), torch.tensor(0.))
                ) / self.batch_sizeK1

        elif self.method == 'distmult':
            ##### DistMult score
            A_loss_matrix = torch.sum(A_rel_batch * A_h_ent_batch * A_t_ent_batch, dim=1)
            A_neg_matrix = torch.sum(A_rel_batch * A_hn_ent_batch * A_tn_ent_batch, dim=1)

            A_loss = torch.sum(
                torch.max(A_neg_matrix - A_loss_matrix + self.m1, torch.tensor(0.))
            ) / self.batch_sizeK1

        elif self.method == 'hole':
            ##### HolE score
            A_loss_matrix = torch.sum(A_rel_batch * circular_correlation(A_h_ent_batch, A_t_ent_batch), dim=1)
            A_neg_matrix = torch.sum(A_rel_batch * circular_correlation(A_hn_ent_batch, A_tn_ent_batch), dim=1)

            A_loss = torch.sum(
                torch.max(A_neg_matrix - A_loss_matrix + self.m1, torch.tensor(0.))
            ) / self.batch_sizeK1

        else:
            raise ValueError('Embedding method not valid!')

        return A_loss

    def forwardB(self, B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index):
        # KG2
        B_h_ent_batch = F.normalize(self.ht2[B_h_index], p=2, dim=1)
        B_t_ent_batch = F.normalize(self.ht2[B_t_index], p=2, dim=1)
        B_rel_batch = self.r2[B_r_index]
        B_hn_ent_batch = F.normalize(self.ht2[B_hn_index], p=2, dim=1)
        B_tn_ent_batch = F.normalize(self.ht2[B_tn_index], p=2, dim=1)

        if self.method == 'transe':
            ##### TransE score
            # This stores h + r - t
            B_loss_matrix = B_h_ent_batch + B_rel_batch - B_t_ent_batch
            # This stores h' + r - t' for negative samples
            B_neg_matrix = B_hn_ent_batch + B_rel_batch - B_tn_ent_batch
            if self.L1:
                B_loss = torch.sum(
                    torch.max(torch.sum(torch.abs(B_loss_matrix), dim=1) + self.m2
                              - torch.sum(torch.abs(B_neg_matrix), dim=1), torch.tensor(0.))
                ) / self.batch_sizeK2
            else:
                B_loss = torch.sum(
                    torch.max(torch.sqrt(torch.sum(torch.square(B_loss_matrix), dim=1)) + self.m2
                              - torch.sqrt(torch.sum(torch.square(B_neg_matrix), dim=1)), torch.tensor(0.))
                ) / self.batch_sizeK2

        elif self.method == 'distmult':
            ##### DistMult score
            B_loss_matrix = torch.sum(B_rel_batch * B_h_ent_batch * B_t_ent_batch, dim=1)
            B_neg_matrix = torch.sum(B_rel_batch * B_hn_ent_batch * B_tn_ent_batch, dim=1)

            B_loss = torch.sum(
                torch.max(B_neg_matrix - B_loss_matrix + self.m2, torch.tensor(0.))
            ) / self.batch_sizeK1

        elif self.method == 'hole':
            ##### HolE score
            B_loss_matrix = torch.sum(B_rel_batch * circular_correlation(B_h_ent_batch, B_t_ent_batch), dim=1)
            B_neg_matrix = torch.sum(B_rel_batch * circular_correlation(B_hn_ent_batch, B_tn_ent_batch), dim=1)

            B_loss = torch.sum(
                torch.max(B_neg_matrix - B_loss_matrix + self.m2, torch.tensor(0.))
            ) / self.batch_sizeK2

        else:
            raise ValueError('Embedding method not valid!')

        return B_loss

    def forwardAM(self, AM_index1, AM_index2, AM_nindex1, AM_nindex2):
        ######################## Type Loss #######################
        AM_ent1_batch = F.normalize(self.ht1[AM_index1], p=2, dim=1)
        AM_ent2_batch = F.normalize(self.ht2[AM_index2], p=2, dim=1)
        AM_ent1_nbatch = F.normalize(self.ht1[AM_nindex1], p=2, dim=1)
        AM_ent2_nbatch = F.normalize(self.ht2[AM_nindex2], p=2, dim=1)

        if self.bridge == 'CG':
            AM_pos_loss_matrix = AM_ent1_batch - AM_ent2_batch
            AM_neg_loss_matrix = AM_ent1_nbatch - AM_ent2_nbatch
        elif self.bridge == 'CMP-linear':
            # c - (W * e + b)
            # AM_pos_loss_matrix = tf.subtract( tf.add(tf.matmul(AM_ent1_batch, M),bias), AM_ent2_batch )
            AM_pos_loss_matrix = F.normalize(torch.matmul(AM_ent1_batch, self.M) + self.b, p=2, dim=1) - AM_ent2_batch
            AM_neg_loss_matrix = F.normalize(torch.matmul(AM_ent1_nbatch, self.M) + self.b, p=2, dim=1) - AM_ent2_nbatch
            # AM_pos_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent1_batch, M), bias), 1), AM_ent2_batch)
            # AM_neg_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent1_nbatch, M), bias), 1), AM_ent2_nbatch)
        elif self.bridge == 'CMP-single':
            # c - \sigma( W * e + b )
            # AM_pos_loss_matrix = tf.subtract( tf.tanh(tf.add(tf.matmul(AM_ent1_batch, M),bias)), AM_ent2_batch )
            AM_pos_loss_matrix = F.normalize(
                torch.tanh(torch.matmul(AM_ent1_batch, self.M) + self.b), p=2, dim=1) - AM_ent2_batch
            AM_neg_loss_matrix = F.normalize(
                torch.tanh(torch.matmul(AM_ent1_nbatch, self.M) + self.b), p=2, dim=1) - AM_ent2_nbatch
            # AM_pos_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent1_batch, M), bias)), 1),
            #                                  AM_ent2_batch)
            # AM_neg_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent1_nbatch, M), bias)), 1),
            #                                  AM_ent2_nbatch)
        elif self.bridge == 'CMP-double':
            # \sigma (W1 * c + bias1) - \sigma(W2 * c + bias1) --> More parameters to be defined
            # AM_pos_loss_matrix = tf.subtract( tf.add(tf.matmul(AM_ent1_batch, Me), b_e), tf.add(tf.matmul(AM_ent2_batch, Mc), b_c))
            # AM_pos_loss_matrix = tf.subtract( tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent1_batch, Me), b_e),1), tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent2_batch, Mc), b_c),1))
            AM_pos_loss_matrix = F.normalize(torch.tanh(torch.matmul(AM_ent1_batch, self.Me) + self.be), p=2, dim=1) \
                                 - F.normalize(torch.tanh(torch.matmul(AM_ent2_batch, self.Mc) + self.bc), p=2, dim=1)
            AM_neg_loss_matrix = F.normalize(torch.tanh(torch.matmul(AM_ent1_nbatch, self.Me) + self.be), p=2, dim=1) \
                                 - F.normalize(torch.tanh(torch.matmul(AM_ent2_nbatch, self.Mc) + self.bc), p=2, dim=1)

            # AM_pos_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent1_batch, Me), b_e)),
            # 1), tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent2_batch, Mc), b_c)), 1)) AM_neg_loss_matrix =
            # tf.subtract(tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent1_nbatch, Me), b_e)), 1),
            # tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent2_nbatch, Mc), b_c)), 1))
        else:
            raise ValueError('Bridge method not valid!')

        # Hinge Loss for AM
        if self.L1:
            AM_loss = torch.sum(torch.max(
                (torch.sum(torch.abs(AM_pos_loss_matrix), dim=1) + self.mA) -
                torch.sum(torch.abs(AM_neg_loss_matrix), dim=1),
                torch.tensor(0.0)
            )) / self.batch_sizeA
        else:
            AM_loss = torch.sum(torch.max(
                (torch.sqrt(torch.sum(torch.square(AM_pos_loss_matrix), dim=1)) + self.mA) -
                torch.sqrt(torch.sum(torch.square(AM_neg_loss_matrix), dim=1)),
                torch.tensor(0.0)
            )) / self.batch_sizeA

        return AM_loss
        # if self.L1:
        #     self._AM_loss = AM_loss = tf.reduce_sum(
        #         tf.maximum(
        #             tf.subtract(tf.add(tf.reduce_sum(tf.abs(AM_pos_loss_matrix), 1), self._mA),
        #                         tf.reduce_sum(tf.abs(AM_neg_loss_matrix), 1)),
        #             0.)) / self._batch_sizeA
        # else:
        #     self._AM_loss = AM_loss = tf.reduce_sum(
        #         tf.maximum(
        #             tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(AM_pos_loss_matrix), 1)), self._mA),
        #                         tf.sqrt(tf.reduce_sum(tf.square(AM_neg_loss_matrix), 1))),
        #             0.)) / self._batch_sizeA
