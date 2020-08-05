import argparse
import sys
sys.path.append('..')

from crisp import Distribution, GibbsPIS, LBPPIS
from scipy.stats import nbinom
from itertools import count


from matplotlib.pyplot import *
from tqdm import tqdm
from random import randrange

my_parser = argparse.ArgumentParser(description='Simple test script to test Gibbs sampling and forward sampling in CRISP.')
my_parser.add_argument('--setup', type=int, required=False, default=2, help="Test setup: 1 or 2")
args = my_parser.parse_args()


print("Using setup {}".format(args.setup))

if args.setup==1:
    def nb_iter(n,p):
        yield 0.0
        nb = nbinom(n,p)
        for i in count():
            pr = nb.pmf(i)
            if pr<1e-5: break
            yield pr

    qEVec = [pr for pr in nb_iter(4,0.6)]
    qIVec = [pr for pr in nb_iter(4,0.5)]

    T = 30
    S = 3

    alpha = 0.001
    beta = 0.01
    p0 = 0.02
    p1 = 0.99

    contacts = [(0,1,17,1),
                (1,0,17,1),
                (2,1,17,1),
                (1,2,17,1),
                (1,2,29,1),
                (2,1,29,1)]

    qEVec = [0.0000000000, 0.05908981283, 0.1656874653, 0.1819578343, 0.154807057,
             0.1198776096, 0.08938884645, 0.06572939883, 0.04819654533,
             0.03543733758, 0.02620080839, 0.01950646727, 0.01463254844,
             0.0110616426, 0.008426626119]

    qIVec = [0.000000000000, 0.000000000000, 0.00000000000, 0.000000000000, 0.000000000000,
             0.0001178655952, 0.0006658439543, 0.002319264193, 0.005825713197, 0.01160465163,
             0.01949056696, 0.02877007836, 0.03842711373, 0.04743309657, 0.05496446107,
             0.06050719418, 0.06386313651, 0.065094874, 0.06444537162, 0.06225794729,
             0.0589104177, 0.05476817903, 0.05015542853, 0.0453410888, 0.04053528452,
             0.03589255717, 0.03151878504, 0.02747963753, 0.02380914891, 0.02051758911,
             0.01759822872, 0.01503287457, 0.0127962154, 0.01085910889, 0.009190974483,
             0.007761463001, 0.006541562648, 0.005504277076]


    tests = [( 0,  9, 0),
             ( 0, 15, 1),
             ( 0, 19, 1),
             ( 2, 29, 1)
             ]

elif args.setup==2:
    qEVec = [0.0000000000, 0.05908981283, 0.1656874653, 0.1819578343, 0.154807057,
             0.1198776096, 0.08938884645, 0.06572939883, 0.04819654533,
             0.03543733758, 0.02620080839, 0.01950646727, 0.01463254844,
             0.0110616426, 0.008426626119]

    qIVec = [0.000000000000, 0.000000000000, 0.00000000000, 0.000000000000, 0.000000000000,
             0.0001178655952, 0.0006658439543, 0.002319264193, 0.005825713197, 0.01160465163,
             0.01949056696, 0.02877007836, 0.03842711373, 0.04743309657, 0.05496446107,
             0.06050719418, 0.06386313651, 0.065094874, 0.06444537162, 0.06225794729,
             0.0589104177, 0.05476817903, 0.05015542853, 0.0453410888, 0.04053528452,
             0.03589255717, 0.03151878504, 0.02747963753, 0.02380914891, 0.02051758911,
             0.01759822872, 0.01503287457, 0.0127962154, 0.01085910889, 0.009190974483,
             0.007761463001, 0.006541562648, 0.005504277076]

    T = 30
    S = 20

    alpha = 0.001
    beta = 0.01
    p0 = 0.05
    p1 = 0.5


    # u=0 has contacts to everybody, except 1
    # u=1 hs no contacts
    # u=2..S-1 have contacts to 0
    contacts = []
    # for i in range(2,S):
    #     contacts.append((0,i,10,1))
    #     contacts.append((i,0,10,1))
    #     contacts.append((0,i,7,1))
    #     contacts.append((i,0,7,1))

    contacts_fwd = []
    for t in range(2,T):
        for i in range(2,S):
            j = i
            while (j == i):
                j = randrange(2,S)
            contacts_fwd.append((j,i,t,1))
            contacts_fwd.append((i,j,t,1))

    contacts_fwd2 = []
    for t in range(2,T):
        for i in range(2,S):
            if (i % 2 == 0):
                j = i + 1
            else:
                j = i - 1
            contacts_fwd2.append((j,i,t,1))
            contacts_fwd2.append((i,j,t,1))

    contacts_lbp = []
    for t in range(2,T):
        for i in range(2,S):
            if (i % 2 == 0):
                j = i + 1
            else:
                j = i - 1
            contacts_lbp.append((j,i,t,1))
            contacts_lbp.append((i,j,t,1))

    tests = []

else:
    raise ValueError("Setup must be 1 or 2")


qE = Distribution([float(q)/sum(qEVec) for q in qEVec])
qI = Distribution([float(q)/sum(qIVec) for q in qIVec])

##########################################################################################
### buildup CRISP many times with repeated calls to advance, thereby forward sampling

Nsamp = 10000
Z = np.zeros((Nsamp, S, T))
tm = time.time()
for i in range(Nsamp):
    ct = [c for c in contacts_fwd if c[2] == 0]
    tt = [o for o in tests if o[1] == 0]

    crisp_fwd = GibbsPIS( S, 1, ct,tt, qE,qI, alpha, beta, p0, p1, True)
    for t in range(1,T):
        ct = [c for c in contacts_fwd if c[2] == t]
        tt = [o for o in tests if o[1] == t]
        crisp_fwd.advance(ct, tt)
    Z[i] = crisp_fwd.sample(N=1)
print("generated {} forward samples in {:.3}s".format(Nsamp, time.time()-tm))
p_fwd = (Z[:,:,:,np.newaxis] == np.arange(4).reshape(1,1,1,-1)).mean(axis=0)

tm = time.time()
for i in range(Nsamp):
    ct = [c for c in contacts_fwd2 if c[2] == 0]
    tt = [o for o in tests if o[1] == 0]

    crisp_fwd = GibbsPIS( S, 1, ct,tt, qE,qI, alpha, beta, p0, p1, True)
    for t in range(1,T):
        ct = [c for c in contacts_fwd2 if c[2] == t]
        tt = [o for o in tests if o[1] == t]
        crisp_fwd.advance(ct, tt)
    Z[i] = crisp_fwd.sample(N=1)
print("generated {} forward samples in {:.3}s".format(Nsamp, time.time()-tm))
p_fwd2 = (Z[:,:,:,np.newaxis] == np.arange(4).reshape(1,1,1,-1)).mean(axis=0)

##########################################################################################
### inference with the LBP model
crisp = LBPPIS(S, T, contacts_lbp, tests,qE,qI, alpha, beta, p0, p1, True)
# crisp.propagate(5,"full")
crisp.propagate(1,"forward")

t = time.time()
p_lbp = crisp.get_marginals()
print("LBP inference in {:.3}s".format(time.time()-t))

##########################################################################################
## plot the results
figure(figsize=(6,7))
plot(p_fwd2[2],'.-')
plot(p_lbp[2],'.--')
grid(True)
title("Forward sampling (social bubbles) and LBP (social bubbles)")

figure(figsize=(6,7))
plot(p_fwd[2],'.-')
plot(p_lbp[2],'.--')
grid(True)
title("Forward sampling (random connections) and LBP (social bubbles)")

show()