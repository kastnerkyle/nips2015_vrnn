import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano
import theano.tensor as T
import sys
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers.cost import biGMMLayer
from cle.cle.cost import biGMM
from cle.cle.utils import unpickle, tolist, OrderedDict, predict
import scipy

from cle.cle.utils.op import logsumexp
from scipy.io import wavfile
from sk.datasets.iamondb import IAMOnDB
from cle.cle.cost import Gaussian, biGMM
from theano.tensor.shared_randomstreams import RandomStreams

#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
#data_path = '/raid/chungjun/data/ubisoft/onomatopoeia/'
#save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/sample/'
#exp_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'

data_path = '/data/lisatmp3/iamondb/'
save_path = '/u/kratarth/Documents/RNN_with_MNIST/NIPS/sk/datasets/handwriting/saves/'
exp_path = '/u/kratarth/Documents/RNN_with_MNIST/NIPS/sk/datasets/handwriting/ground_truth/nips/'

frame_size = 3
# How many samples to generate
# How many examples you want to proceed at a time
batch_size = 719
debug = 1

exp_name = 'm3_1'

train_data = IAMOnDB(name='train',
                   path=data_path,
                   prep = 'normalize')

X_mean = train_data.X_mean
X_std = train_data.X_std

valid_data = IAMOnDB(name='valid',
                   prep = 'normalize',
                   path=data_path,
                   X_mean=X_mean,
                   X_std=X_std)

x, x_mask = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)
    temp = np.ones((15, batch_size), dtype=np.float32)
    temp[:, -2:] = 0.
    x_mask.tag.test_value = temp

exp = unpickle(exp_path + exp_name + '_best.pkl')
nodes = exp.model.nodes
names = [node.name for node in nodes]
num_sample = 40

main_lstm, prior, kl,\
x_1,\
z_1, \
phi_1, phi_mu, phi_sig, \
prior_1, prior_mu, prior_sig, \
theta_1,  theta_mu, theta_sig, coeff, corr, binary = nodes


def inner_fn(x_t, s_tm1, s_tm1_is):

    x_1_t = x_1.fprop([x_t])
    
    phi_emb_1_t = phi_1.fprop([x_1_t, s_tm1])

    phi_mu_t = phi_mu.fprop([phi_emb_1_t])
    phi_sig_t = phi_sig.fprop([phi_emb_1_t])

    prior_emb_1_t = prior_1.fprop([s_tm1])

    prior_mu_t = prior_mu.fprop([prior_emb_1_t])
    prior_sig_t = prior_sig.fprop([prior_emb_1_t])

    z_t = prior.fprop([phi_mu_t, phi_sig_t])
    kl_t = kl.fprop([phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t])

    z_emb_1_t = z_1.fprop([z_t])

    theta_1_t = theta_1.fprop([z_emb_1_t, s_tm1])
    theta_mu_t = theta_mu.fprop([theta_1_t])
    theta_sig_t = theta_sig.fprop([theta_1_t])
    coeff_t = coeff.fprop([theta_1_t])
    corr_t = corr.fprop([theta_1_t])
    binary_t = binary.fprop([theta_1_t])

    s_t = main_lstm.fprop([[x_1_t, z_emb_1_t], [s_tm1]])

    x_t_is = T.repeat(x_t, num_sample, axis=0)
    x_1_t_is = x_1.fprop([x_t_is])
 
    phi_1_t_is = phi_1.fprop([x_1_t_is, s_tm1_is])
    phi_mu_t_is = phi_mu.fprop([phi_1_t_is])
    phi_sig_t_is = phi_sig.fprop([phi_1_t_is])

    prior_1_t_is = prior_1.fprop([s_tm1_is])
    prior_mu_t_is = prior_mu.fprop([prior_1_t_is])
    prior_sig_t_is = prior_sig.fprop([prior_1_t_is])

    z_t_is = prior.sample([phi_mu_t_is, phi_sig_t_is])
    z_1_t_is = z_1.fprop([z_t_is])

    theta_1_t_is = theta_1.fprop([z_1_t_is, s_tm1_is])
    theta_mu_t_is = theta_mu.fprop([theta_1_t_is])
    theta_sig_t_is = theta_sig.fprop([theta_1_t_is])
    coeff_t_is = coeff.fprop([theta_1_t_is])
    corr_t_is = corr.fprop([theta_1_t_is])
    binary_t_is = binary.fprop([theta_1_t_is])
    mll = biGMM(x_t_is, theta_mu_t_is, theta_sig_t_is, coeff_t_is, corr_t_is, binary_t_is) +\
          Gaussian(z_t_is, prior_mu_t_is, prior_sig_t_is) -\
          Gaussian(z_t_is, phi_mu_t_is, phi_sig_t_is)
    mll = mll.reshape((batch_size, num_sample))
    mll = logsumexp(-mll, axis=1) - T.log(num_sample)

    s_t_is = main_lstm.fprop([[x_1_t_is, z_1_t_is], [s_tm1_is]])

    return s_t, s_t_is, kl_t, theta_mu_t, theta_sig_t, coeff_t, corr_t, binary_t, mll


((s_t,s_t_is,kl_t, theta_mu_t, theta_sig_t, coeff_t, corr_t, binary_t, mll), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[main_lstm.get_init_state(batch_size),
                main_lstm.get_init_state(batch_size*num_sample),
                              None, None, None, None, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_theta_sig = theta_sig_t.reshape((theta_sig_t.shape[0]*theta_sig_t.shape[1], -1))
reshaped_coeff = coeff_t.reshape((coeff_t.shape[0]*coeff_t.shape[1], -1))
reshaped_corr = corr_t.reshape((corr_t.shape[0]*corr_t.shape[1], -1))
reshaped_binary = binary_t.reshape((binary_t.shape[0]*binary_t.shape[1], -1))

recon = biGMM(reshaped_x, reshaped_theta_mu, reshaped_theta_sig, reshaped_coeff, reshaped_corr, reshaped_binary)
recon = recon.reshape((theta_mu_t.shape[0], theta_mu_t.shape[1]))
recon = recon * x_mask
recon_term = recon.sum(axis=0).mean()
kl_t = kl_t * x_mask
kl_term = kl_t.sum(axis=0).mean()

nll_lower_bound = recon_term + kl_term
nll_lower_bound.name = 'nll_lower_bound'

mll = mll * x_mask
mll = -mll.sum(axis=0).mean()
mll.name = 'marginal_nll'

outputs = [mll, nll_lower_bound]
monitor_fn = theano.function(inputs=[x, x_mask],
                     outputs=outputs,
                     on_unused_input='ignore',
                     allow_input_downcast=True)

DataProvider = [Iterator(valid_data, batch_size)]

data_record = []
for data in DataProvider:
    batch_record = []
    for batch in data:
        this_out = monitor_fn(*batch)
        batch_record.append(this_out)
    data_record.append(np.asarray(batch_record))
for record, data in zip(data_record, DataProvider):
    for i, ch in enumerate(outputs):
        this_mean = record[:, i].mean()
        if this_mean is np.nan:
            raise ValueError("NaN occured in output.")
        print("%s_%s: %f" % (data.name, ch.name, this_mean))

