import numpy as np
import matplotlib.pyplot as plt


def create_game_data(game_num, turn_num, H_prior):
    H_values = [0, 1, 2, 3, 5, 7, 9]
    H = np.random.choice(H_values, game_num, p=H_prior)
    d = np.random.choice(range(1,7), (game_num, turn_num))
    O = np.mod(d * H[:, np.newaxis], 10)
    return H_values, H, O

def h_probability(o_string):
    p_s = np.zeros(shape=(10, 7))
    p = np.zeros(shape=(H_num, o_string.size+1)) +1
    H = np.array([0, 1, 2, 3, 5, 7, 9])
    results = (H[np.newaxis] * d_temp[:, np.newaxis]) % 10
    p_s=o_probabilty(results)
    for i in range(1, o_string.size + 1, 1):
        for h in range(7):
            p[h,i] = p[h,i-1]*p_s[o_string[i-1]][h]
            deno = 0
            for h_dummy in range(H_num):
                deno += p[h_dummy,i-1]*p_s[o_string[i-1],h_dummy]
            p[h,i] /= deno
    max=0
    argmax=0
    for h in range (H_num):
        if(p[h,o_string.size]>max):
            argmax=h
            max=p[h,o_string.size]
    return H[argmax]
def o_probabilty(results):

    s=np.zeros(shape=(10,7))
    for h in range(7):
        for i in range(10):
            for d in range(6):
                if(results[d, h]==i):
                    s[i,h]+=1
            s[i, h] /= 6

    return(s)

def tester(guess_func, game_num, turn_num, H_prior):
    assert len(H_prior.shape) == 1
    assert len(H_prior) == 7

    # create games data
    H_values, H, O = create_game_data(game_num, turn_num, H_prior)
    H_num = len(H_values)
    d_num = 6

    # play games
    Hhat = np.zeros(game_num)
    posteriors = np.zeros((game_num, H_num))
    for j in range(game_num):
        Hhat[j] = guess_func(O[j])

    # compute error rate for each value of H
    erred = Hhat != H
    error_rate = np.zeros(H_num)
    for k, h in enumerate(H_values):
        error_rate[k] = np.mean(erred[H == h])

    # show error rates
    plt.figure()
    plt.plot(H_values, error_rate, '.')
    plt.xlabel('H value')
    plt.ylabel('Error rate')
    plt.show()



#create P(O|H)
H_num = 7
d_num = 6
d_temp = np.arange(1,d_num+1,1)
prior_H = np.ones(H_num) / H_num
tester(h_probability,1000,20,prior_H)

#TODO: remake the o by given h table again, i messed it up. it needs to be in h_p func.