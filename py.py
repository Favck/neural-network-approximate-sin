import numpy as np


def predict(x):
    W1 = np.load("W1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")
    b1 = np.load("b1.npy")

    z1 = W1 @ x + b1
    a1 = 1/(1+np.exp(-z1))
    pred = W2 @ a1 + b2
    return pred

class Layer_in:

    def __init__(self):
        self.W1 = np.random.normal(size=(3, 1))
        self.b1 = np.random.normal(size=(3, 1))

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def forward(self, x):
        self.z1 = (self.W1 @ x).reshape(-1,1) + self.b1
        np.save("z1.npy", self.z1)
        self.a1 = self.sigmoid(self.z1)
        np.save("a1.npy", self.a1)
        return self.a1

    def update_W1(self, W1):
        self.W1 = W1
        np.save("W1", self.W1)
        return True

    def update_b1(self, b1):
        self.b1 = b1
        np.save("b1", self.b1)


class Layer1(Layer_in):

    W2 = np.random.normal(size=(1, 3))
    b2 = np.random.normal(size=(1, 1))


    def sigmoid_derivatives(self, z):
        return np.exp(-z)/((1 + np.exp(-z))**2)
    
    def forward(self):

        self.z2 = self.W2 @ np.load("a1.npy") + self.b2
        
        np.save("z2.npy", self.z2)
        return self.z2
    
    def backward(self, target):
        der1 = (2*(np.load("z2.npy") - target) * self.W2.T)
        
        der2 = self.sigmoid_derivatives(np.load("z1.npy"))
        der3 = der1 * der2

        super().update_W1(1 - 0.001*(der3 @ np.load("a1.npy").T))
        super().update_b1(der1)
    
    def update_W2(self, W2):
        self.W2 = W2
        np.save("W2", self.W2)

    def update_b2(self, b2):
        self.b2 = b2
        np.save("b2", self.b2)
        

class Layer_out(Layer1):



    def out(self):
        self.y = np.load("z2.npy")
        
        return self.y

    def backward(self, target):
        der1 = (2*(np.load("z2.npy") - target) @ np.load("a1.npy").T)

        super().update_W2(1 - 0.001*der1)
        super().update_b2(2*(np.load("z2.npy") - target))

    def loss(self, target):
        L = (self.out() - target)**2
        return L



if __name__=="__main__":

    data = np.array([[x, np.sin(x)] for x in range(-1000, 1000)])

    data_train = data[:1800]
    data_test = data[1800:]

    x_train = np.array([x[0] for x in data_train])
    y_train = np.array([y[1] for y in data_train])

    print(predict(np.array([0])))



    # N = 2

    # Lay_in = Layer_in()
    # Lay1 = Layer1()
    # Lay_out = Layer_out()
    
    # for epoch in range(N):
    #     Loss = []
    #     np.random.shuffle(x_train)
    #     np.random.shuffle(y_train)
    #     for i in range(0, data_train.shape[0]):
    #         x_batch = x_train[i:i+32].reshape(-1, 1)
    #         y_batch = y_train[i:i+32].reshape(-1, 1)
    #         loss_batch = 0
    #         for x, y in zip(x_batch, y_batch):
    #             x = np.array([x])
    #             y = np.array([y])
    #             Lay_in.forward(x)
    #             Lay1.forward()
    #             Lay_out.out()

    #             loss_batch += Lay_out.loss(y)

    #             Lay1.backward(y)
    #             Lay_out.backward(y)
    #         Loss.append(loss_batch)
    #         print(f'Loss {epoch} epoch: {loss_batch/32}  sqrt: {np.sqrt(loss_batch/32)}')

    
    #     print(f'Loss {epoch} epoch: {sum(Loss)/len(Loss)}  sqrt: {np.sqrt(sum(Loss)/len(Loss))}')
            

    






