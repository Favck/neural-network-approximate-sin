import numpy as np
import matplotlib.pyplot as plt
import torch



def forward_pass(x, W, b):
    z = x @ W.T + b

    return z

def update_weight(w, b):
    w.data -= 0.001*w.grad.data
    b.data -= 0.001*b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()
    return w, b

def plot_sin(x, y, w, b, fig, ax, line_true, line_pred):
    x_ = x.view(-1, 1)          
    y_ = y.view(-1, 1)          

    idx = torch.argsort(x_.squeeze())

    with torch.no_grad():
        x_norm = x_ / (3 * np.pi)   
        z = forward_pass(x_norm, w[0], b[0])
        a = torch.tanh(z)
        y_pred = forward_pass(a, w[1], b[1])

    x_np      = x_[idx].numpy()
    y_np      = y_[idx].numpy()
    y_pred_np = y_pred[idx].cpu().numpy()

    line_true.set_data(x_np, y_np)
    line_pred.set_data(x_np, y_pred_np)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


if __name__=="__main__":


    x_raw = torch.linspace(-3*np.pi, 3*np.pi, 5000, dtype=torch.float32)
    x = x_raw/(3*np.pi)
    y = torch.sin(x_raw)
    

    D_in, H, D_out = 1, 512, 1

    w1 = torch.randn(H, D_in, requires_grad=True, dtype=torch.float32)
    w2 = torch.randn(D_out, H, requires_grad=True, dtype=torch.float32)
    b1 = torch.randn(H, requires_grad=True, dtype=torch.float32)
    b2 = torch.randn(D_out,requires_grad=True, dtype=torch.float32)

    Epoch = 2000
    batch = 64
    plt.ion()
    fig, ax = plt.subplots()

    line_true, = ax.plot([], [], 'b', label='true')
    line_pred, = ax.plot([], [], 'r', label='pred')

    ax.legend()
    ax.set_xlim(-10*np.pi, 10*np.pi)
    ax.set_ylim(-1.5, 1.5)
    history = []
    for epoch in range(Epoch):
        
        LOSS = 0.0
        idx = torch.randperm(x.size(0))
        x_shuf = x[idx]
        y_shuf = y[idx]
        for i in range(0, x.shape[0], batch):
            x_batch = x_shuf[i:i+batch].view(-1,1)
            y_batch = y_shuf[i:i+batch].view(-1,1)
            
            
            z1 = forward_pass(x_batch, w1, b1)
            a1 = torch.tanh(z1)

            y_pred = forward_pass(a1, w2, b2)
            
            loss = torch.mean((y_pred - y_batch)**2)
            LOSS += loss.item()
            history.append(loss.item())
            loss.backward()

            w1, b1 = update_weight(w1, b1)
            w2, b2 = update_weight(w2, b2)

            if i%2000 == 0:
                plot_sin(x_raw, y, (w1, w2), (b1, b2), fig, ax, line_true, line_pred)



        print(f"{epoch+1}/{Epoch} LOSS:{LOSS/(x.shape[0])}")
    
    plt.ioff() 
    plt.figure() 
    plt.plot(history[:1000])
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss value")
    plt.show()



