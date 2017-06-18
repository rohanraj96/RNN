import numpy as np

data=open('input.txt','r').read()
chars=list(set(data))
data_size,vocab_size=len(data),len(chars)
char_to_index={ch:i for i,ch in enumerate(chars)}
index_to_char={i:ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_size=100
seq_length=25
learning_rate=1e-1

# model parameters
Wxh=0.01*np.random.randn(hidden_size,vocab_size)
Whh=0.01*np.random.randn(hidden_size,hidden_size)
bh=np.zeros((hidden_size,1))
Why=0.01*np.random.randn(vocab_size,hidden_size)
by=np.zeros((vocab_size,1))

def lossfun(inputs,targets,hprev):
    xs,hs,ys,ps={},{},{},{}
    hs[-1]=np.copy(hprev)
    loss=0
    # forward prop
    for t in xrange(len(inputs)):
        xs[t]=np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        hs[t]=np.tanh(np.dot(Whh,hs[t-1])+bh + np.dot(Wxh,xs[t]))
        ys[t]=np.dot(Why,hs[t])+by
        ps[t]=np.exp(ys[t])/np.sum(np.exp(ys[t]))
        loss+= -np.log(ps[t][targets[t],0])
    dWxh,dWhh,dWhy=np.zeros_like(Wxh),np.zeros_like(Whh),np.zeros_like(Why)
    dbh,dby=np.zeros_like(bh),np.zeros_like(by)
    dhnext=np.zeros_like(hs[0])
    # backward prop
    for t in reversed(xrange(len(inputs))):
        dy=np.copy(ps[t])
        dy[targets[t]]-=1
        dWhy+=np.dot(dy,hs[t].T)
        dby+=dy
        dh=np.dot(Why.T,dy)+dhnext
        dhraw=(1-hs[t]*hs[t])*dh
        dbh+=dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]

def sample(h,seed_ix,n):
    x=np.zeros((vocab_size,1))
    print x.shape
    x[seed_ix]=1
    ixes=[]
    for t in xrange(n):
        h=np.tanh(np.dot(Wxh,x)+np.dot(Whh,h)+bh)
        y=np.dot(Why,h)+by
        p=np.exp(y)/np.sum(np.exp(y))
        ix=np.random.choice(range(vocab_size),p=p.ravel())
        x=np.zeros((vocab_size,1))
        x[ix]=1
        ixes.append(ix)
    return ixes

n,p=0,0
# Memory variables for adagrad update
mWxh,mWhh,mWhy=np.zeros_like(Wxh),np.zeros_like(Whh),np.zeros_like(Why)
mbh,mby=np.zeros_like(bh),np.zeros_like(by)
smooth_loss=-np.log(1.0/vocab_size)*seq_length
while True:
    if p+seq_length>=len(data) or n==0:
        hprev=np.zeros((hidden_size,1))
        p=0
    inputs=[char_to_index[ch] for ch in data[p:p+seq_length]]
    targets=[char_to_index[ch] for ch in data[p+1:p+seq_length+1]]
    if n%100==0:
        sample_ix=sample(hprev,inputs[0],200)
        txt=''.join(index_to_char[ix] for ix in sample_ix)
        print '----\n %s \n----' % (txt, )
    loss,dWxh,dWhh,dWhy,dbh,dby,hprev=lossfun(inputs,targets,hprev)
    smooth_loss=smooth_loss*0.999+loss*0.001
    if n%100==0:
        print "iter = %d loss = %f"%(n,smooth_loss)
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
                                mem += dparam * dparam
                                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    p+=seq_length
    n+=1
