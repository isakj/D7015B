#%%

import numpy as np
import pandas as pd
import math

import matplotlib
import matplotlib.pyplot as plt

#%%
# read the data file into a dataframe
df = pd.read_csv('machine_data.csv')

print(df.shape)
print(dict(df).keys())

#%%
"""
Extract data for a given manufacturer
"""
grpByManu = df.groupby(['manufacturef'])

df_g = {}

df_g['A'] = grpByManu.get_group('A')
df_g['B'] = grpByManu.get_group('B')
df_g['C'] = grpByManu.get_group('c') # for some reason, manufacturer c is lowercase in data

manufacturers = [ 'A', 'B', 'C' ]

sum = 0
for manufacturer in manufacturers:
    print('Manufacturer {} has {} data points'.format(manufacturer, df_g[manufacturer].shape[0]))
    sum += df_g[manufacturer].shape[0]

assert sum == df.shape[0]


#%%

# load and time by manufacturer
load_m = {}
time_m = {}

for manufacturer in manufacturers:
    load_m[manufacturer] = df_g[manufacturer]['load']
    time_m[manufacturer] = df_g[manufacturer]['time']


#%%
'''
Characteristics of data
range, mean, median, stddev
'''
for manufacturer in manufacturers:
    for column in ['load','time']:
        print('Range of {} for manufacturer {}: {:.2f} - {:.2f}'.format(column, manufacturer, 
                                                                        df_g[manufacturer][column].min(),
                                                                        df_g[manufacturer][column].max()))
        print('Mean of {} for manufacturer {}: {:.2f}'.format(column, manufacturer, df_g[manufacturer][column].mean()))
        print('Median of {} for manufacturer {}: {:.2f}'.format(column, manufacturer, df_g[manufacturer][column].median()))
        print('Stddev of {} for manufacturer {}: {:.2f}'.format(column, manufacturer, np.std(df_g[manufacturer][column])))

#%%
'''
Characteristics of data
samples vs stddev
'''
print('  ', end='')
for manufacturer in manufacturers:
    print('{:40s}'.format(manufacturer), end='')
print()
print('  '.format(''), end='')
for manufacturer in manufacturers:
    for column in ['load','time']:
        print('{:20s}'.format(column), end='')
print()
print('  '.format(''), end='')
for manufacturer in manufacturers:
    for column in ['load','time']:
        print('{:10s}{:10s}'.format('expected','actual'), end='')
print()
for sigma in range(1,7):
    print('{:<2d}'.format(sigma), end='')
    for manufacturer in manufacturers:
        for column in ['load','time']:
            d = df_g[manufacturer][column]
            mean = d.mean()
            std = np.std(d)
            print('{:7.3f} % {:7.3f} % '.format(
                math.erf(sigma/math.sqrt(2))*100.0,
                ((d > mean-sigma*std) & (d < mean+sigma*std)).sum() * 100.0 / d.count()
            ), end='')
    print()

#%%
'''
How is load distributed
Why does it matter
uniform, normal, exponential, weibull
'''
figure, axis = plt.subplots(2, 3, layout='tight')
m = 0
for manufacturer in manufacturers:
    load = load_m[manufacturer]
    time = time_m[manufacturer]
    axis[0,m].hist(load, bins=10)
    axis[0,m].set_title(manufacturer)
    axis[0,m].set_xlabel('load')
    axis[0,m].set_ylabel('frequency')
    axis[1,m].hist(time, bins=10)
    axis[1,m].set_xlabel('time')
    axis[1,m].set_ylabel('frequency')
    m += 1
plt.savefig('assignment2-distribution.png',dpi=300)
plt.show()

#%%
'''
Is there a relationship between load and time
'''
polyline = np.linspace(65, 85, 100) 
for degree in range(1,4):
    t = 'Polynomial regression, $n={}$\n'.format(degree)
    for manufacturer in manufacturers:
        load = load_m[manufacturer]
        time = time_m[manufacturer]
        # plot sample data
        plt.scatter(load, time, label=manufacturer)
        # try a polynomial fit
        fit,ssr,_,_,_ = np.polyfit(load, time, degree, full=True)
        sstot = np.var(time)*load.shape[0]
        R2 = 1 - ssr[0]/sstot
        t += '$R_{}^2$={:.8f}\n'.format(manufacturer, R2)
        plt.plot(polyline, np.poly1d(fit)(polyline))

    plt.axis([64,86,0,60])
    plt.text(73,35,t)
    plt.legend()
    plt.title("Relation between load and time")
    plt.xlabel("load")
    plt.ylabel("time")
    plt.savefig('assignment2-regression-{}.png'.format(degree),dpi=300)
    plt.show()

