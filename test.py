import numpy as np

def dropout(x, level):
    if level < 0. or level >= 1  :
        raise Exception('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level
    sample =np.random.binomial(n=1 ,p=retain_prob ,size=x.shape  )
    print(sample)
    x *= sample
    print(x)
    x /= retain_prob

    return x

x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
dropout(x, 0.1)