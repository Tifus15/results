import torch
import dgl
from ghnn_model import *
import matplotlib.pyplot as plt

model3dof = torch.load("res_3dof/model.pt").cpu() # roll
model4dof = torch.load("res_4dof/model.pt").cpu() # GHNN


print(model3dof)
print(model4dof)

x3 = torch.load("res_3dof/eval.pt")
x4 = torch.load("res_4dof/eval.pt")

h3 = torch.load("res_3dof/eval_H.pt")
h4 = torch.load("res_4dof/eval_H.pt")

losses3 = torch.load("res_3dof/losses.pt").transpose(0,1)
losses4 = torch.load("res_3dof/losses.pt").transpose(0,1)
print(losses3.shape)
print(x3.shape)
print(h4.shape)

epochs_t = torch.linspace(1,losses3.shape[0],losses3.shape[0])



fig, ax  = plt.subplots(1,4)
ax[0].set_title("Train/Test GHNN")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")

ax[1].set_title("roll")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")

ax[2].set_title("vec")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")

ax[3].set_title("h")
ax[3].set_xlabel("epochs")
ax[3].set_ylabel("loss")






ax[0].semilogy(epochs_t,losses3[:,0],c="b")
ax[0].semilogy(epochs_t,losses3[:,4],c="r")

ax[1].semilogy(epochs_t,losses3[:,1],c="b")
ax[1].semilogy(epochs_t,losses3[:,5],c="r")

ax[2].semilogy(epochs_t,losses3[:,2],c="b")
ax[2].semilogy(epochs_t,losses3[:,6],c="r")

ax[3].semilogy(epochs_t,losses3[:,3],c="b")
ax[3].semilogy(epochs_t,losses3[:,7],c="r")




ax[0].legend(["train loss","test_loss"])
ax[1].legend(["train loss","test_loss"])
ax[2].legend(["train loss","test_loss"])
ax[3].legend(["train loss","test_loss"])

fig.suptitle("3dof case")




fig, ax  = plt.subplots(1,4)
ax[0].set_title("Train/Test Loss GRUGHNN")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")

ax[1].set_title("roll")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")

ax[2].set_title("vec")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")

ax[2].set_title("h")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")





ax[0].semilogy(epochs_t,losses4[:,0],c="b")
ax[0].semilogy(epochs_t,losses4[:,4],c="r")

ax[1].semilogy(epochs_t,losses4[:,1],c="b")
ax[1].semilogy(epochs_t,losses4[:,5],c="r")

ax[2].semilogy(epochs_t,losses4[:,2],c="b")
ax[2].semilogy(epochs_t,losses4[:,6],c="r")

ax[3].semilogy(epochs_t,losses4[:,3],c="b")
ax[3].semilogy(epochs_t,losses4[:,7],c="r")

ax[0].legend(["train loss","test_loss"])
ax[1].legend(["train loss","test_loss"])
ax[2].legend(["train loss","test_loss"])
ax[3].legend(["train loss","test_loss"])


fig.suptitle("4dof case")

plt.show()
