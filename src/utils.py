# some useful tools
import numpy as np
# from numpy.fft import fft, ifft

# def circular_correlation(h, t):
#     return tf.real(tf.spectral.ifft(tf.multiply(tf.conj(tf.spectral.fft(tf.complex(h, 0.))), tf.spectral.fft(tf.complex(t, 0.)))))
#
# def np_ccorr(h, t):
#     return ifft(np.conj(fft(h)) * fft(t)).real
#
# def make_hparam_string(dim, onto_ratio, type_ratio, lr):
# 	# input params: dim, onto_ratio, type_ratio, lr,
# 	return "dim_%s_onto_%s_type_%s_lr_%.0E" % (dim, onto_ratio, type_ratio,lr)

import torch
import torch.fft as fft

def circular_correlation(h, t):
    fft_h = fft.fft(torch.complex(h, torch.zeros_like(h)))
    fft_t = fft.fft(torch.complex(t, torch.zeros_like(t)))
    ccorr = fft.ifft(torch.conj(fft_h) * fft_t)
    return ccorr.real

def np_ccorr(h, t):
    fft_h = torch.fft.fft(h)
    fft_t = torch.fft.fft(t)
    ccorr = torch.fft.ifft(torch.conj(fft_h) * fft_t)
    return ccorr.real.numpy()

def make_hparam_string(dim, onto_ratio, type_ratio, lr):
    return f"dim_{dim}_onto_{onto_ratio}_type_{type_ratio}_lr_{lr:.0E}"