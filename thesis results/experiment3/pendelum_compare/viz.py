import torch
import dgl
from ghnn_model import *
import matplotlib.pyplot as plt

model31 = torch.load("model31.pt").cpu() # roll
model32 = torch.load("model32.pt").cpu() # GHNN
model41 = torch.load("model41.pt").cpu() # roll
model42 = torch.load("model42.pt").cpu()# GHNN

print(model31)
print(model32)
print(model41)
print(model42)

a3 = torch.load("eval3.pt")
a4 = torch.load("eval4.pt")
losses = torch.load("losses.pt").transpose(0,1)
print(losses.shape)
print(a3.shape)
print(a4.shape)

epochs_t = torch.linspace(1,losses.shape[0],losses.shape[0])



fig, ax  = plt.subplots(2,4)
ax[0,0].set_title("Train/Test Loss GRUGHNN")
ax[0,0].set_xlabel("epochs")
ax[0,0].set_ylabel("loss")

ax[0,1].set_title("roll")
ax[0,1].set_xlabel("epochs")
ax[0,1].set_ylabel("loss")

ax[0,2].set_title("vec")
ax[0,2].set_xlabel("epochs")
ax[0,2].set_ylabel("loss")

ax[0,3].set_title("h")
ax[0,3].set_xlabel("epochs")
ax[0,3].set_ylabel("loss")

ax[1,0].set_title("Train/Test Loss GHNN")
ax[1,0].set_xlabel("epochs")
ax[1,0].set_ylabel("loss")

ax[1,1].set_title("roll")
ax[1,1].set_xlabel("epochs")
ax[1,1].set_ylabel("loss")

ax[1,2].set_title("vec")
ax[1,2].set_xlabel("epochs")
ax[1,2].set_ylabel("loss")

ax[1,3].set_title("h")
ax[1,3].set_xlabel("epochs")
ax[1,3].set_ylabel("loss")




ax[0,0].semilogy(epochs_t,losses[:,0],c="b")
ax[0,0].semilogy(epochs_t,losses[:,16],c="r")

ax[0,1].semilogy(epochs_t,losses[:,1],c="b")
ax[0,1].semilogy(epochs_t,losses[:,17],c="r")

ax[0,2].semilogy(epochs_t,losses[:,2],c="b")
ax[0,2].semilogy(epochs_t,losses[:,18],c="r")

ax[0,3].semilogy(epochs_t,losses[:,3],c="b")
ax[0,3].semilogy(epochs_t,losses[:,19],c="r")

ax[1,0].semilogy(epochs_t,losses[:,4],c="b")
ax[1,0].semilogy(epochs_t,losses[:,20],c="r")

ax[1,1].semilogy(epochs_t,losses[:,5],c="b")
ax[1,1].semilogy(epochs_t,losses[:,21],c="r")

ax[1,2].semilogy(epochs_t,losses[:,6],c="b")
ax[1,2].semilogy(epochs_t,losses[:,22],c="r")

ax[1,3].semilogy(epochs_t,losses[:,7],c="b")
ax[1,3].semilogy(epochs_t,losses[:,23],c="r")


ax[0,0].legend(["train loss","test_loss"])
ax[0,1].legend(["train loss","test_loss"])
ax[0,2].legend(["train loss","test_loss"])
ax[0,3].legend(["train loss","test_loss"])

ax[1,0].legend(["train loss","test_loss"])
ax[1,1].legend(["train loss","test_loss"])
ax[1,2].legend(["train loss","test_loss"])
ax[1,3].legend(["train loss","test_loss"])
fig.suptitle("3dof case")




fig, ax  = plt.subplots(2,4)
ax[0,0].set_title("Train/Test Loss GRUGHNN")
ax[0,0].set_xlabel("epochs")
ax[0,0].set_ylabel("loss")

ax[0,1].set_title("roll")
ax[0,1].set_xlabel("epochs")
ax[0,1].set_ylabel("loss")

ax[0,2].set_title("vec")
ax[0,2].set_xlabel("epochs")
ax[0,2].set_ylabel("loss")

ax[0,3].set_title("h")
ax[0,3].set_xlabel("epochs")
ax[0,3].set_ylabel("loss")

ax[1,0].set_title("Train/Test Loss GHNN")
ax[1,0].set_xlabel("epochs")
ax[1,0].set_ylabel("loss")

ax[1,1].set_title("roll")
ax[1,1].set_xlabel("epochs")
ax[1,1].set_ylabel("loss")

ax[1,2].set_title("vec")
ax[1,2].set_xlabel("epochs")
ax[1,2].set_ylabel("loss")

ax[1,3].set_title("h")
ax[1,3].set_xlabel("epochs")
ax[1,3].set_ylabel("loss")




ax[0,0].semilogy(epochs_t,losses[:,8],c="b")
ax[0,0].semilogy(epochs_t,losses[:,24],c="r")

ax[0,1].semilogy(epochs_t,losses[:,9],c="b")
ax[0,1].semilogy(epochs_t,losses[:,25],c="r")

ax[0,2].semilogy(epochs_t,losses[:,10],c="b")
ax[0,2].semilogy(epochs_t,losses[:,26],c="r")

ax[0,3].semilogy(epochs_t,losses[:,11],c="b")
ax[0,3].semilogy(epochs_t,losses[:,27],c="r")

ax[1,0].semilogy(epochs_t,losses[:,12],c="b")
ax[1,0].semilogy(epochs_t,losses[:,28],c="r")

ax[1,1].semilogy(epochs_t,losses[:,13],c="b")
ax[1,1].semilogy(epochs_t,losses[:,29],c="r")

ax[1,2].semilogy(epochs_t,losses[:,14],c="b")
ax[1,2].semilogy(epochs_t,losses[:,30],c="r")

ax[1,3].semilogy(epochs_t,losses[:,15],c="b")
ax[1,3].semilogy(epochs_t,losses[:,31],c="r")


ax[0,0].legend(["train loss","test_loss"])
ax[0,1].legend(["train loss","test_loss"])
ax[0,2].legend(["train loss","test_loss"])
ax[0,3].legend(["train loss","test_loss"])

ax[1,0].legend(["train loss","test_loss"])
ax[1,1].legend(["train loss","test_loss"])
ax[1,2].legend(["train loss","test_loss"])
ax[1,3].legend(["train loss","test_loss"])
fig.suptitle("4dof case")

plt.show()