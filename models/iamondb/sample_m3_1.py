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
from cle.cle.layers.feedforward import FullyConnectedLayer2
from cle.cle.utils import unpickle, tolist, OrderedDict, predict
import scipy

from scipy.io import wavfile
from sk.datasets.iamondb import IAMOnDB

from theano.tensor.shared_randomstreams import RandomStreams

#Don't use a python long as this don't work on 32 bits computers.
np.random.seed(0xbeef)
rng = RandomStreams(seed=np.random.randint(1 << 30))
#data_path = '/raid/chungjun/data/ubisoft/onomatopoeia/'
#save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/sample/'
#exp_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'

data_path = '/data/lisatmp3/iamondb/'
save_path = '/u/kastner/m3_samples/'
exp_path = '/data/lisatmp3/kratarth/iamondb/'

frame_size = 3
# How many samples to generate
batch_size = 10
# How many timesteps to generate
n_steps = 700
debug = 1

exp_name = 'm3_1'
save_name = 'm3_1_sample'

data = IAMOnDB(name='train',
                          prep = 'normalize',
                          path=data_path,
                          bias = 0)


exp = unpickle(exp_path + exp_name + '_best.pkl')
nodes = exp.model.nodes
names = [node.name for node in nodes]
bias = T.lscalar()
output = biGMMLayer(name='output',
                  parent=['theta_mu',
                          'theta_sig',
                          'coeff',
                          'corr',
                          'binary' ],
                  use_sample=1,
                  nout=frame_size-1)
#p_x_1_dim = 250
#k = 20
#coeff2 = FullyConnectedLayer2(name='coeff',
#                             parent=['theta_emb_1'],
#                             parent_dim=[p_x_1_dim],
#                              nout=k,
#                             unit='softmax')

coder, prior, kl,\
x_emb_1,\
z_emb_1,\
phi_emb_1,phi_mu, phi_sig,\
prior_emb_1, prior_mu, prior_sig,\
theta_emb_1, theta_mu, theta_sig, coeff, corr, binary = nodes
#import ipdb;ipdb.set_trace()
#coeff2.params = coeff.params

def inner_fn(s_tm1, bias):

    prior_emb_1_t = prior_emb_1.fprop([s_tm1])

    prior_mu_t = prior_mu.fprop([prior_emb_1_t])
    prior_sig_t = prior_sig.fprop([prior_emb_1_t])

    z_t = prior.fprop([prior_mu_t, prior_sig_t])

    z_emb_1_t = z_emb_1.fprop([z_t])

    theta_emb_1_t = theta_emb_1.fprop([z_emb_1_t, s_tm1])

    theta_mu_t = theta_mu.fprop([theta_emb_1_t])
    theta_sig_t = theta_sig.fprop([theta_emb_1_t])
    theta_sig_t = T.log(T.exp(theta_sig_t) -1. ) - bias
    theta_sig_t = T.nnet.softplus(theta_sig_t)
    coeff_t = coeff.fprop([theta_emb_1_t])
    #coeff_t = T.log(coeff_t)
    #coeff_t -= T.mean(coeff_t, axis=1).dimshuffle(0,'x')
    #coeff_t = coeff_t * (1. + bias)
    #coeff_t = T.nnet.softmax(coeff_t)
    #coeff2_t = coeff2.fprop([theta_emb_1_ti], (1.+bias))
    corr_t = corr.fprop([theta_emb_1_t])
    binary_t = binary.fprop([theta_emb_1_t])
    binary_t = rng.binomial(size=binary_t.shape, n=1, p=binary_t,
                                dtype=theano.config.floatX)
    '''
    mu = theta_mu_t
    sig = theta_sig_t
    mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff_t.shape[-1],
                         coeff_t.shape[-1]))
    sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff_t.shape[-1],
                           coeff_t.shape[-1]))
    idx = predict(
            rng.multinomial(
                pvals=coeff_t,
                dtype=coeff_t.dtype
            ),
            axis=1
    )
    mu = mu[T.arange(mu.shape[0]), :, idx]
    sig = sig[T.arange(sig.shape[0]), :, idx]
    corr_t = corr_t[T.arange(corr_t.shape[0]), idx]

    mu_x = mu[:,0]
    mu_y = mu[:,1]
    sig_x = sig[:,0]
    sig_y = sig[:,1]

    z = rng.normal(size=mu.shape,
                   avg=0., std=1.,
                   dtype=mu.dtype)
    s_x = (mu_x + sig_x * z[:,0]).dimshuffle(0,'x')
    s_y = (mu_y + sig_y * ( (z[:,0] * corr_t) + (z[:,1] * T.sqrt(1.-corr_t**2) ) )).dimshuffle(0,'x')
    x_t = T.concatenate([binary_t,s_x,s_y], axis = 1)
    mu_t = mu
    '''
    #ipdb.set_trace()
    x_t,mu_t = output.sample_mean([theta_mu_t, theta_sig_t, coeff_t, corr_t, binary_t])

    x_emb_1_t = x_emb_1.fprop([x_t])

    s_t = coder.fprop([[x_emb_1_t, z_emb_1_t], [s_tm1]])

    return s_t, x_t, T.concatenate([binary_t,mu_t], axis =1)

((s_t, y, m), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[coder.get_init_state(batch_size),
                              None, None],
                non_sequences = bias,
                n_steps=n_steps)

for k, v in updates.iteritems():
    k.default_update = v

test_fn = theano.function(inputs=[bias],
                          outputs=[y, m],
                          updates=updates,
                          allow_input_downcast=True,
                          on_unused_input='ignore')

samples = test_fn(data.bias)[0]
print samples.shape
samples = np.transpose(samples, (1, 0, 2))

samples[:,:,1:] = (samples[:,:,1:] * data.X_std) + data.X_mean
samples[:,:,1:] = np.cumsum(samples[:,:,1:], axis =1)

start = 0
all_strokes = []
for n in range(samples.shape[0]):
    all_strokes.append([])
    penup = np.where(samples[n, :, 0] == 1.)[0]
    sample = samples[n, :, 1:]
    for pu in penup:
        all_strokes[n].append(sample[start:pu])
        start = pu + 1

for n, _ in enumerate(all_strokes):
    ymin = np.concatenate(all_strokes[n], axis=0)[:, 1].min()
    ymax = np.concatenate(all_strokes[n], axis=0)[:, 1].max()
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for s in all_strokes[n]:
        # turning antialiasing on makes lines smoother
        plt.plot(s[:, 0], s[:, 1], color="k", linewidth=1, antialiased=False)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(save_path + save_name +'_black_'+str(n) + '.png')
    plt.clf()

# make final image with
# for i in *.png; do convert $i -gravity Center -crop 80x20%+0+0 $i; done
# convert *.png -append stacked.png
