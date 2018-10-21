import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_name = ['new_data/lander_nq.pkl', 'new_data/lander_dq.pkl']#,'new_data/exp_atari_nd.pkl']
#file_name = ['new_data/pong_nd.pkl','new_data/exp_atari_nd.pkl','new_data/atari_bay.pkl']
data = []
for i in file_name:
	with open(i, 'rb') as handle:
		a = pickle.load(handle)
		# Take 4m data
		print(len(a))
		#a = a[:400]
		#print((np.asarray(a)).shape)
		#print(type(a))
		data.append(a)

ndata = np.asarray(data)
print(ndata.shape)

print(ndata.shape)

sns.tsplot(time = ndata[0,:,0], data = ndata[0,:,1], color = 'r', linestyle = '-', condition = 'nd_average')
sns.tsplot(time = ndata[0,:,0], data = ndata[0,:,2], color = 'r', linestyle = '--', condition = 'nd_best')
sns.tsplot(time = ndata[1,:,0], data = ndata[1,:,1], color = 'g', linestyle = '-', condition = 'dq_average')
sns.tsplot(time = ndata[1,:,0], data = ndata[1,:,2], color = 'g', linestyle = '--', condition = 'dq_best')
#sns.tsplot(time = ndata[2,:,0], data = ndata[2,:,1], color = 'b', linestyle = '-', condition = 'Bay_average')
#sns.tsplot(time = ndata[2,:,0], data = ndata[2,:,2], color = 'b', linestyle = '--', condition = 'Bay_best')

plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Vanilla v.s. Double Q Learning on LunarLander")

plt.legend(loc="lower right")
plt.show()
plt.show()
