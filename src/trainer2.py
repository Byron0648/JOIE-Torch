import numpy as np
import time
import torch
import torch.nn as nn

from multiG import multiG
import model2 as model


class Trainer(object):
    def __init__(self, multiG, method='transe', bridge='CG-one', dim1=300, dim2=50, batch_sizeK1=1024,
                 batch_sizeK2=1024,
                 batch_sizeA=32, a1=5., a2=0.5, m1=0.5, m2=1.0, save_path='this-model.ckpt',
                 multiG_save_path='this-multiG.bin', L1=False, lr=0.001, w=1.0, AM_fold=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.multiG = multiG
        self.method = method
        self.bridge = bridge
        self.dim1 = self.multiG.dim1 = self.multiG.KG1.dim = dim1  # update dim
        self.dim2 = self.multiG.dim2 = self.multiG.KG2.dim = dim2  # update dim
        self.batch_sizeK1 = self.multiG.batch_sizeK1 = batch_sizeK1
        self.batch_sizeK2 = self.multiG.batch_sizeK2 = batch_sizeK2
        self.batch_sizeA = self.multiG.batch_sizeA = batch_sizeA
        self.a1 = a1
        self.a2 = a2
        self.m1 = m1
        self.m2 = m2
        self.save_path = save_path
        self.multiG_save_path = multiG_save_path
        self.L1 = self.multiG.L1 = L1
        self.model = model.joie(num_rels1=self.multiG.KG1.num_rels(),
                                num_ents1=self.multiG.KG1.num_ents(),
                                num_rels2=self.multiG.KG2.num_rels(),
                                num_ents2=self.multiG.KG2.num_ents(),
                                method=self.method,
                                bridge=self.bridge,
                                dim1=dim1,
                                dim2=dim2,
                                batch_sizeK1=self.batch_sizeK1,
                                batch_sizeK2=self.batch_sizeK2,
                                batch_sizeA=self.batch_sizeA,
                                L1=self.L1)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.m1 = m1
        self.model.m2 = m2
        self.w = w
        self.AM_fold = AM_fold
        self.device = device

    def gen_KM_batch(self, KG_index, batchsize, forever=False, shuffle=True):  # batchsize is required
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        l = KG.triples.shape[0]
        while True:
            triples = KG.triples
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, batchsize):
                batch = triples[i: i + batchsize, :]
                if batch.shape[0] < batchsize:
                    batch = np.concatenate((batch, self.multiG.triples[:batchsize - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == batchsize
                neg_batch = KG.corrupt_batch(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield torch.from_numpy(h_batch.astype(np.int64)), torch.from_numpy(r_batch.astype(np.int64)), torch.from_numpy(t_batch.astype(np.int64)), torch.from_numpy(neg_h_batch.astype(
                    np.int64)), torch.from_numpy(neg_t_batch.astype(np.int64))
            if not forever:
                break

    def gen_AM_batch(self, forever=False, shuffle=True):  # not changed with its batchsize
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i + self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                n_batch = multiG.corrupt_align_batch(batch, tar=1)  # only neg on class
                e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch[:, 0], batch[:, 1], n_batch[:, 0], n_batch[:, 1]
                yield torch.from_numpy(e1_batch.astype(np.int64)), torch.from_numpy(e2_batch.astype(np.int64)), torch.from_numpy(e1_nbatch.astype(
                    np.int64)), torch.from_numpy(e2_nbatch.astype(np.int64))
            if not forever:
                break

    def gen_AM_batch_non_neg(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i + self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                e1_batch, e2_batch = batch[:, 0], batch[:, 1]
                yield torch.from_numpy(e1_batch.astype(np.int64)), torch.from_numpy(e2_batch.astype(np.int64))
            if not forever:
                break

    def train1epoch_KM(self, num_A_batch, num_B_batch, epoch):

        this_gen_A_batch = self.gen_KM_batch(KG_index=1, batchsize=self.batch_sizeK1, forever=True)
        this_gen_B_batch = self.gen_KM_batch(KG_index=2, batchsize=self.batch_sizeK2, forever=True)

        km_total_loss = 0.0
        for batch_id in range(num_A_batch):
            # Optimize loss A
            A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index = next(this_gen_A_batch)
            A_h_index.to(self.device)
            A_r_index.to(self.device)
            A_hn_index.to(self.device)
            A_tn_index.to(self.device)
            self.optim.zero_grad()
            loss_A = self.model.forwardA(A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index)
            loss_A.backward()
            self.optim.step()
            km_total_loss += loss_A.item()
            if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):
                print('\rprocess KG1: %d / %d. Epoch %d' % (batch_id + 1, num_A_batch + 1, epoch))

        for batch_id in range(num_B_batch):
            # Optimize loss B
            B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index = next(this_gen_B_batch)
            B_h_index.to(self.device)
            B_r_index.to(self.device)
            B_hn_index.to(self.device)
            B_tn_index.to(self.device)
            self.optim.zero_grad()
            loss_B = self.model.forwardB(B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index)
            loss_B.backward()
            self.optim.step()
            # Observe total loss
            km_total_loss += loss_B.item()
            if ((batch_id + 1) % 500 == 0 or batch_id == num_B_batch - 1):
                print('\rprocess KG2: %d / %d. Epoch %d' % (batch_id + 1, num_B_batch + 1, epoch))

        print("KM Loss of epoch", epoch, ":", km_total_loss)
        return km_total_loss

    def train1epoch_AM(self, num_AM_batch, epoch):

        this_gen_AM_batch = self.gen_AM_batch(forever=True)
        # this_gen_AM_batch = self.gen_AM_batch_non_neg(forever=True)

        am_total_loss = 0.0

        for batch_id in range(num_AM_batch):
            # Optimize loss A
            AM_index1, AM_index2, AM_nindex1, AM_nindex2 = next(this_gen_AM_batch)
            AM_index1.to(self.device)
            AM_index2.to(self.device)
            AM_nindex1.to(self.device)
            AM_nindex2.to(self.device)
            self.optim.zero_grad()
            loss_AM = self.model.forwardAM(AM_index1, AM_index2, AM_nindex1, AM_nindex2)
            loss_AM.backward()
            self.optim.step()
            # Observe total loss
            am_total_loss += loss_AM.item()
            if ((batch_id + 1) % 100 == 0) or batch_id == num_AM_batch - 1:
                print('\rprocess: %d / %d. Epoch %d' % (batch_id + 1, num_AM_batch + 1, epoch))

        print("AM Loss of epoch", epoch, ":", am_total_loss)
        return am_total_loss

    def train1epoch_associative(self, epoch):

        num_A_batch = int(self.multiG.KG1.num_triples() / self.batch_sizeK1)
        num_B_batch = int(self.multiG.KG2.num_triples() / self.batch_sizeK2)
        num_AM_batch = int(self.multiG.num_align() / self.batch_sizeA)

        if epoch <= 1:
            print('num_KG1_batch =', num_A_batch)
            print('num_KG2_batch =', num_B_batch)
            print('num_AM_batch =', num_AM_batch)
        loss_KM = self.train1epoch_KM(num_A_batch, num_B_batch, epoch)
        # keep only the last loss
        for i in range(self.AM_fold):
            loss_AM = self.train1epoch_AM(num_AM_batch, epoch)
        return loss_KM, loss_AM

    def train(self, epochs=20, save_every_epoch=10):
        self.model.to(self.device)
        t0 = time.time()
        for epoch in range(epochs):
            ## 这个写法好像w还是没有用，实际上
            epoch_lossKM, epoch_lossAM = self.train1epoch_associative(epoch)
            loss = epoch_lossKM + self.w * epoch_lossAM
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_lossKM) or np.isnan(epoch_lossAM):
                print("Training collapsed.")
                return
            if (epoch + 1) % save_every_epoch == 0:
                torch.save(self.model.state_dict(), self.save_path)
                # this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
                self.multiG.save(self.multiG_save_path)
                print("MTransE saved in file: %s. Multi-graph saved in file: %s" % (
                    self.save_path, self.multiG_save_path))
        torch.save(self.model.state_dict(), self.save_path)
        # this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
        self.multiG.save(self.multiG_save_path)
        print("MTransE saved in file: %s. Multi-graph saved in file: %s" % (self.save_path, self.multiG_save_path))
        print("Done")
