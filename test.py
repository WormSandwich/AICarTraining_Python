import numpy as np 

x = np.array([[10, 9, 12, 13]])
print(x.argmax(axis=1)[0])

smax = x[0][0]
sindex = 0


print('Smax {0:} {1:}'.format(sindex, smax))

for index,value in enumerate(x[0]):
    print('{0:} {1:}'.format(index, value))
    if value>smax and value < x[0][x.argmax(axis=1)[0]]:
        smax = value
        sindex = index

print('Second {0:} {1:}'.format(sindex, smax))

