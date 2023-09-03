import numpy as np
import matplotlib.pyplot as plt

# Parameters
p_values = np.array([0.02, 0.2, 0.6])
n_iter = int(1e4)
n_tests = int(1e2)
n_gr = 10

# Initialize arrays
tests = np.empty((p_values.size, n_iter, n_tests))
pooled_tests = np.zeros((p_values.size, n_iter, n_gr))

# Simulate tests
for i, p in enumerate(p_values):
    tests[i, :, :] = np.random.choice(2, (n_iter, n_tests), p=[1 - p, p])

# Pooling tests
k = n_tests // n_gr
for j in range(n_gr):
    pooled_tests[:, :, j] = np.sum(tests[:, :, k * j:k * (j + 1)], axis=-1)

pooled_tests[pooled_tests > 0] = k

X = np.sum(pooled_tests + 1, axis=-1)

Ex_X_emp = np.mean(X, axis=-1)

# Theoretical calculation
Ex_X = n_tests + n_tests / k - n_tests * (1 - p_values) ** k

print('Theoretical expectation: ' + str(Ex_X))
print('Empirical expectation: ' + str(Ex_X_emp))

# Find optimal integer k
k_values = np.arange(1, 101)
f_k_values = []

for p in p_values:
    f_k = 1 + 1 / k_values - ((1 - p) ** k_values)
    f_k_values.append(f_k)

# Plot f(k) for different values of p
fig = plt.figure(figsize=[15, 5])
for i, p in enumerate(p_values):
    plt.plot(k_values, f_k_values[i], label='p = ' + str(p))

plt.xlabel('k')
plt.ylabel('f(k)')
plt.legend()
plt.title('f(k) for different values of p')
plt.show()


p = np.linspace(0.01,0.5,int(1e4))

k = (np.arange(100)+1)
f = 1 + 1/k - np.power.outer((1-p),k)
# Condition for pooling to be better is f_k < 1

k_opt = k[np.argmin(f,axis = 1 )]

f_opt = 1 + 1/k_opt - np.power((1-p),k_opt)

fig = plt.figure(figsize = [15,5])
plt.plot(p,f_opt)
plt.plot(p,np.ones(p.shape),'--')
plt.plot([0.25,0.25],[np.amin(f_opt),np.amax(f_opt)],'--')
plt.xlabel('p')
plt.ylabel('f_opt')
plt.legend(['f_opt','Condition for pooling', 'Theoretical pooling limit'])
plt.title('f_opt for different values of p')
plt.show()