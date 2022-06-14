# ==========================================
#           GENERAL FUNCTIONS
# ==========================================
import h5py
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
#              QUESTION 1
# ==========================================

class AutoEncoder:
    def __init__(self, Lin, Lhid, lmbda, beta, rho):
        self.Lin = Lin
        self.Lhid = Lhid
        self.lmbda = lmbda
        self.beta = beta
        self.rho = rho
        
    def initializeLayers(self, N):
        self.hidden = np.zeros((self.Lhid,N))
        
    def initializeWeights(self):
        #Initialization described in the prompt
        w0  = np.sqrt(6/(self.Lin+self.Lhid))
        self.weights = np.random.uniform(-w0,w0,(self.Lhid,self.Lin))
        self.bIH = np.random.uniform(-w0,w0,self.Lhid)
        self.bHO = np.random.uniform(-w0,w0,self.Lin)
        self.JgradW_ho, self.JgradW_ih, self.Jgradb_ho, self.Jgradb_ih = 0, 0, 0, 0
    
    def forward(self, data):
        self.A1 = np.dot(self.weights,data.T)+self.bIH[:,None]
        self.hidden = sigmoid(self.A1)
        self.p_b = np.mean(self.hidden,axis=1)[:,None]
        self.A2 = np.dot(self.weights.T,self.hidden)+self.bHO[:,None]
        self.out = sigmoid(self.A2).T
        return self.out
    
    def calculateCost(self, data, batchCount):
        meanSquaredError = np.square(data-self.out).mean()/2
        tykhonov = (np.sum(np.square(self.weights))+np.sum(np.square(self.weights.T)))*self.lmbda/2
        KL = self.rho*np.log(self.rho/self.p_b) + (1-self.rho)*np.log((1-self.rho)/(1-self.p_b))
        self.J = meanSquaredError + tykhonov/batchCount + self.beta*np.sum(KL)
        return self.J, meanSquaredError, tykhonov/batchCount, self.beta*np.sum(KL)
        
    def calculateGrad(self, data, momentum, batchCount):
        # Calculates the gradient of each sample than sums it up. forward() should be run before
        N = data.shape[0]
        gradWT, gradW, gradb1, gradb2 = 0,0,0,0
        for i in range(N):
            outAvg = self.out[i][:,None]
            hidAvg = self.hidden[:,i][:,None]
            datAvg = data[i][:,None]
            dJ1_do__do_dA2 = (outAvg-datAvg)*(outAvg*(1-outAvg))/N
            T3 = self.beta*((1-self.rho)/(1-hidAvg)-self.rho/hidAvg)*(1/N)
            gradWT = gradWT + dJ1_do__do_dA2*hidAvg.T
            gradW = gradW + (np.sum(dJ1_do__do_dA2*(self.weights.T),axis=0).reshape((self.Lhid,1))+T3)*(hidAvg*(1-hidAvg))*datAvg.T
            gradb2 = gradb2 + dJ1_do__do_dA2
            gradb1 = gradb1 + (np.sum(dJ1_do__do_dA2*(self.weights.T),axis=0).reshape((self.Lhid,1))+T3)*(hidAvg*(1-hidAvg))
        self.JgradW_ho = gradWT + momentum*self.JgradW_ho + self.lmbda*self.weights.T/batchCount
        self.JgradW_ih = gradW + momentum*self.JgradW_ih + self.lmbda*self.weights/batchCount
        self.Jgradb_ho = gradb2 + momentum*self.Jgradb_ho
        self.Jgradb_ih = gradb1 + momentum*self.Jgradb_ih
        
    def updateWeights(self, LR):
        self.weights = self.weights - LR*self.JgradW_ih - LR*self.JgradW_ho.T
        self.bIH = self.bIH - LR*self.Jgradb_ih.reshape(self.Lhid,)
        self.bHO = self.bHO - LR*self.Jgradb_ho.reshape(self.Lin,)
        
def trainMiniQ1(encoder, data, epoch, learningRate, momentum, batchSize = 32):
    lossList = []
    N = data.shape[0]
    batchCount = N//batchSize
    remainder = N % batchSize
    remLimit = N - remainder
    for e in range(epoch):
        permutation = np.random.permutation(N)# shuffle the data before each epoch
        shuffled_samples = data[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        if remainder != 0:
            samples.append(shuffled_samples[remLimit:])
            print("yes")
        loss = 0
        for j in range(len(samples)):
            # each method is run sequentially
            encoder.forward(samples[j])
            loss += np.array(encoder.calculateCost(samples[j], len(samples)))
            encoder.calculateGrad(samples[j], momentum, len(samples))
            encoder.updateWeights(learningRate)
        lossList.append(np.trunc(loss*10**6)/(10**6))
        print(f"Loss [Total, MSE, Tykhonov, KL] in epoch {e+1}: {lossList[e]}")
    return lossList, samples


def normalizeGray(arrayIn):
    # Done sequentially according to the prompt
    gray_scale = 0.2126*arrayIn[:,0,:] + 0.7152*arrayIn[:,0,:] + 0.0722*arrayIn[:,0,:]
    mean = gray_scale.mean(axis=1)
    gray_scale = gray_scale - mean[:,None]
    std = gray_scale.std()
    clipped = np.clip(gray_scale, -3*std, 3*std)
    minG, maxG = np.min(clipped), np.max(clipped)
    normalizedOut = (clipped-minG)*4/(maxG-minG)/5 + 0.1
    return normalizedOut


def plotArrayGray(arrayIn, rows, columns, offset=0):
    fig = plt.figure(figsize=(2*rows,1.2*columns))
    minA, maxA = np.min(arrayIn), np.max(arrayIn) # find min and max points so that the images are normalized
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(arrayIn[i+offset-1].reshape((16,16)), cmap='gray', vmin=minA, vmax=maxA)
        plt.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def plotArrayRGB(arrayIn, rows, columns, offset=0):
    fig = plt.figure(figsize=(2*rows,1.2*columns))
    minA, maxA = np.min(arrayIn), np.max(arrayIn)
    arrayIn = arrayIn.reshape((columns*rows,3,16,16))
    arrayIn = [[[tuple(row) for row in xdim] for xdim in np.moveaxis(instance, 0, -1)] for instance in arrayIn]
    # Reshapes the array so that it can be plotted
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(arrayIn[i+offset-1], vmin=minA, vmax=maxA)
        plt.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

def plotParameterQ1(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [str(i) for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with {metricName[2]} Hidden Neurons, Lambda: {metricName[3]}, Beta: {metricName[4]}, Rho: {metricName[5]} Learning Rate: {metricName[6]}, Momentum: {metricName[7]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# ==========================================
#              QUESTION 2
# ==========================================
import h5py
import numpy as np
import matplotlib.pyplot as plt

class fourgramNN:
    def __init__(self, Lembed, Lhid, Lfeature):
        self.Lembed = Lembed
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        
    def initializeLayers(self, N):
        self.embed1 = np.zeros((self.Lembed, N))
        self.embed2 = np.zeros((self.Lembed, N))
        self.embed3 = np.zeros((self.Lembed, N))
        self.hidden = np.zeros((self.Lhid, N))
        self.out = np.zeros((self.Lfeature, N))
        
    def initializeWeights(self):
        self.weights1 = np.random.normal(0, 0.01, (self.Lembed, self.Lfeature))
        self.weights21 = np.random.normal(0, 0.01, (self.Lhid, self.Lembed))
        self.weights22 = np.random.normal(0, 0.01, (self.Lhid, self.Lembed))
        self.weights23 = np.random.normal(0, 0.01, (self.Lhid, self.Lembed))
        self.weights3 = np.random.normal(0, 0.01, (self.Lfeature, self.Lhid))
        self.b2 = np.random.normal(0, 0.01, (self.Lhid,1))
        self.b3 = np.random.normal(0, 0.01, (self.Lfeature,1))
        
        self.gradW1, self.gradW21, self.gradW22, self.gradW23, self.gradW3, self.gradb2, self.gradb3 = 0,0,0,0,0,0,0
        self.gradW1New, self.gradW21New, self.gradW22New, self.gradW23New, self.gradW3New, self.gradb2New, self.gradb3New = 0,0,0,0,0,0,0
    
    
    def forward(self, data):
        # idxToOH is used as it can't process the not normalized data
        self.embed1 = np.dot(self.weights1,idxToOH(data[0]))
        self.embed2 = np.dot(self.weights1,idxToOH(data[1]))
        self.embed3 = np.dot(self.weights1,idxToOH(data[2]))
        A11 = np.dot(self.weights21,self.embed1)+self.b2
        A12 = np.dot(self.weights22,self.embed2)+self.b2
        A13 = np.dot(self.weights23,self.embed3)+self.b2
        self.hidden = sigmoid(A11+A12+A13)
        A2 = np.dot(self.weights3,self.hidden)+self.b3
        self.out = softmax(A2).T # make the shape same with data
        return self.out
    
    def forwardWord(self):
        return np.argmax(self.out,axis=1) + 1
    
    def crossEntropy(self, ground):
        return -np.log(self.out.T[ground-1])
        
    def calculateGrad(self, data, ground):
        #wrong gradient
        out = self.out
        hid = self.hidden
        em1 = self.embed1
        em2 = self.embed2
        em3 = self.embed3
        #dJ_do__do_dA22 = (-(1/out[gnd-1])*(np.diag(self.out[i])-out*out.T))[:,gnd-1][:,None] long way to do the same
        dJ_do__do_dA2 = out.T - idxToOH(ground)
        dJ_dA1 = np.dot(self.weights3.T,dJ_do__do_dA2)*(hid*(1-hid))
        dJ_dE1 = np.dot(self.weights21.T,dJ_dA1)
        dJ_dE2 = np.dot(self.weights22.T,dJ_dA1)
        dJ_dE3 = np.dot(self.weights23.T,dJ_dA1)
        gradW3 = dJ_do__do_dA2*hid.T
        gradW21 = dJ_dA1*em1.T
        gradW22 = dJ_dA1*em2.T
        gradW23 = dJ_dA1*em3.T
        gradW1 = (dJ_dE1*idxToOH(data[0]).T + dJ_dE2*idxToOH(data[1]).T + dJ_dE3*idxToOH(data[2]).T)/3
        gradb3 = dJ_do__do_dA2
        gradb2 = dJ_dA1
        self.gradW1New = self.gradW1New + gradW1
        self.gradW21New = self.gradW21New + gradW21
        self.gradW22New = self.gradW22New + gradW22
        self.gradW23New = self.gradW23New + gradW23
        self.gradW3New = self.gradW3New + gradW3
        self.gradb2New = self.gradb2New + gradb2
        self.gradb3New = self.gradb3New + gradb3
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        self.gradW1 = momentum*self.gradW1 + self.gradW1New
        self.gradW21 = momentum*self.gradW21 + self.gradW21New
        self.gradW22 = momentum*self.gradW22 + self.gradW22New
        self.gradW23 = momentum*self.gradW23 + self.gradW23New
        self.gradW3 = momentum*self.gradW3 + self.gradW3New
        self.gradb2 = momentum*self.gradb2 + self.gradb2New
        self.gradb3 = momentum*self.gradb3 + self.gradb3New
        
        self.weights1 = self.weights1 - learningRate*self.gradW1
        self.weights21 = self.weights21 - learningRate*self.gradW21
        self.weights22 = self.weights22 - learningRate*self.gradW22
        self.weights23 = self.weights23 - learningRate*self.gradW23
        self.weights3 = self.weights3 - learningRate*self.gradW3
        self.b2 = self.b2 - learningRate*self.gradb2
        self.b3 = self.b3 - learningRate*self.gradb3
        
        self.gradW1New, self.gradW21New, self.gradW22New, self.gradW23New, self.gradW3New, self.gradb2New, self.gradb3New = 0,0,0,0,0,0,0
    
    
    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target.reshape(target.shape[1]))
        self.calcGrad(sample, target.reshape(target.shape[1]))
        return loss, guess

def idxToOH(idx, word = 250):
    oneHot = np.zeros((word,1))
    oneHot[idx-1] = 1
    return oneHot

def trainMiniBatchQ2(nnModel, data, ground, valdat, valgnd, epoch, learningRate, momentum, batchSize = 200):
    # similar to the mini batch trainer for Q1 but uses validation
    lossListT, lossListV = [], []
    totalSamples = len(ground)
    batchCount = totalSamples//batchSize
    remainder = totalSamples % batchSize
    remLimit = totalSamples - remainder
    for e in range(epoch):
        permutation = list(np.random.permutation(totalSamples))
        shuffled_samples = data[permutation]
        shuffled_grounds = ground[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        grounds = np.array_split(shuffled_grounds[:remLimit], batchCount)
        samples.append(shuffled_samples[remLimit:])
        grounds.append(shuffled_grounds[remLimit:])
        loss = 0
        for j in range(len(grounds)):
            bSize = len(grounds[j])
            for i in range(bSize):
                nnModel.forward(samples[j][i])
                loss += nnModel.crossEntropy(grounds[j][i])
                nnModel.calculateGrad(samples[j][i], grounds[j][i])
            nnModel.updateWeights(learningRate,momentum)
        lossListT.append(loss)
        loss = 0
        for i in range(len(valgnd)):
            nnModel.forward(valdat[i])
            loss += nnModel.crossEntropy(valgnd[i])
        loss = loss/len(valgnd)
        lossListV.append(loss)
        print(f"Training and Validation Loss in epoch {e+1}: {lossListT[e]}, {lossListV[e]}")
        
        if loss > 1.2*lossListT[0]: 
            print("Terminated due to increased loss")
            return lossListV, lossListT
        elif (e > 1) & (lossListT[e-1] - lossListT[e] < 0.01):
            print("Terminated due to convergence")
            return lossListV, lossListT
    return lossListV, lossListT

def estimateForward(model, testX, testD, words, best=10):
    # estimates the words for given model, wordlist test input and ground
    for i in range(len(testD)):
        probs = model.forward(testX[i])
        idx = probs.argsort()[0,-best:][::-1]
        s = testX[i]
        gnd = testD[i]
        print(f"\nTrigram: {words[s[0]]} {words[s[1]]} {words[s[2]]} [{words[gnd]}], Guess: ",end="")
        for j in range(best):
            print(words[idx[j]+1],end=" ")
    print()
    return probs, idx

def plotParameterQ2(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [str(i) for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with {metricName[2]} Embed Size, {metricName[3]} Hidden Neurons, Learning Rate: {metricName[4]}, Momentum: {metricName[5]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# In[]
# ==========================================
#              QUESTION 3
# ==========================================
def plotParameter(metric, labels, metricName):
    # All parameter plotters run the same code but their titles are different.
    plt.figure(figsize = (12,6))
    xlabel = [str(i) for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with Learning Rate: {metricName[2]}, Momentum: {metricName[3]}, BPTT: {metricName[4]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotConf(mat_con, Title):
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            px.text(x=m,y=n,s=int(mat_con[m, n]), va='center', ha='center', size='xx-large')
    # Sets the labels
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix for '+Title, fontsize=15)
    plt.show()

def comp_confmat(actual, predicted):
    np.seterr(divide='ignore')
    classes = np.unique(actual)
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return confmat 


def trainMiniBatch(nnModel, data, ground, valX, valD, testX, testD, epoch, learningRate, momentum, batchSize = 32):
    # very similar to the mini batch trainer in Q2 but utilizes test sets and measures its accuracy.
    # Used by Q3.A Q3.B Q3.C
    countSamples = 0
    lossListT, lossListV, accuracyListT, accTest= [], [], [], []
    totalSamples = len(ground)
    batchCount = totalSamples//batchSize
    remainder = totalSamples % batchSize
    remLimit = totalSamples - remainder
    for e in range(epoch):
        permutation = list(np.random.permutation(totalSamples))
        shuffled_samples = data[permutation]
        shuffled_grounds = ground[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        grounds = np.array_split(shuffled_grounds[:remLimit], batchCount)
        samples.append(shuffled_samples[remLimit:])
        grounds.append(shuffled_grounds[remLimit:])
        
        estimatesT = []
        loss = 0
        for j in range(len(grounds)):
            bSize = grounds[j].shape[0]
            for i in range(bSize):
                countSamples += 1
                l, g = nnModel.trainStep(samples[j][i], grounds[j][i][None,:])
                estimatesT.append(g)
                loss += l
            nnModel.updateWeights(learningRate, momentum)
        loss = loss/totalSamples
        lossListT.append(loss)
        
        gndidx = np.array([np.where(r==1)[0][0] for r in shuffled_grounds]) + 1
        estidx = np.array([np.where(r==1)[0][0] for r in estimatesT]) + 1
        
        falses = np.count_nonzero(gndidx-estidx)
        accuracy = 1-falses/totalSamples
        accuracyListT.append(accuracy)
        
        loss = 0
        for i in range(valD.shape[0]):
            nnModel.forward(valX[i])
            guess, _ = nnModel.forwardOut()
            loss += nnModel.crossEntropy(valD[i][None,:])
        loss = loss/valD.shape[0]
        lossListV.append(loss)
        
        estTest = []
        for i in range(testD.shape[0]):
            nnModel.forward(testX[i])
            guess, _ = nnModel.forwardOut()
            estTest.append(guess)
        
        Tgndidx = np.array([np.where(r==1)[0][0] for r in testD]) + 1
        estTestidx = np.array([np.where(r==1)[0][0] for r in estTest]) + 1
        
        falses = np.count_nonzero(Tgndidx-estTestidx)
        accuracy = 1-falses/testD.shape[0]
        accTest.append(accuracy)
        
        print(f"Validation Loss in epoch {e+1}: {loss}, Test Accuracy: {accuracy}")
        if loss > 1.2*lossListV[0]: 
            print("Termnated due to increased loss")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
        elif (e > 1) & (lossListT[e-1] - lossListT[e] < 0.0001):
            print("Terminated due to convergence")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
    return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)

# ==========================================
#                Q3.A
class RNNLayer:
        # Single RNN layer for single timestep. Stores variables,
        # and calculates gradient with stored variables.
        # forward() should be called before backward() so that
        # internal variables are set.
    def forward(self, x, oldS, U, W, V):
        self.mulUX = np.dot(U, x)
        self.mulWos = np.dot(W, oldS)
        self.add = self.mulWos + self.mulUX
        self.s = np.tanh(self.add) #activation, s = hidden
        self.mulV = np.dot(V, self.s) #out
        
    def backward(self, x, oldS, U, W, V, sDiff, dmulv):
        # calculate the derivatives step by step
        dV = np.asarray(np.dot(np.asmatrix(dmulv).T, np.asmatrix(self.s))) #write this better, shorter
        dSv = np.dot(V.T,dmulv)
        ds = dSv + sDiff
        dadd = (1-np.square(np.tanh(self.add)))*ds
        dmulUX = dadd * np.ones_like(self.mulUX)
        dmulWos = dadd * np.ones_like(self.mulWos)
        dW = np.asarray(np.dot(np.asmatrix(dmulWos).T, np.asmatrix(oldS)))
        doldS = np.dot(W.T,dmulWos)
        dU = np.asarray(np.dot(np.asmatrix(dmulUX).T, np.asmatrix(x)))
        return doldS, dU, dW, dV


class RNN:
    def __init__(self, Lfeature, Lxdim, Lhid, Lclass, bptt):
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        self.Lxdim = Lxdim
        self.Lclass = Lclass
        self.bptt = bptt
        
    def initializeWeights(self):
        # Initialize with Xavier
        self.U = np.random.uniform(-np.sqrt(1./self.Lxdim),np.sqrt(1./self.Lxdim),(self.Lhid,self.Lxdim))#weights between input and hidden layers
        self.W = np.random.uniform(-np.sqrt(1./self.Lhid),np.sqrt(1./self.Lhid),(self.Lhid,self.Lhid))
        self.V = np.random.uniform(-np.sqrt(1./self.Lhid),np.sqrt(1./self.Lhid),(self.Lclass,self.Lhid))
        self.dU = np.zeros(self.U.shape)
        self.dW = np.zeros(self.W.shape)
        self.dV = np.zeros(self.V.shape)
        self.dUnew = np.zeros(self.U.shape)
        self.dWnew = np.zeros(self.W.shape)
        self.dVnew = np.zeros(self.V.shape)
    
    def forward(self, data):
        #data is (T,) size timeseries.
        T = len(data)
        foldedLayers = []
        oldS = np.zeros(self.Lhid)
        for t in range(T):
                rnnlayer = RNNLayer()
                rnnlayer.forward(data[t], oldS, self.U, self.W, self.V)
                oldS = rnnlayer.s
                foldedLayers.append(rnnlayer)
        self.rnnLayers = foldedLayers
        return foldedLayers
    
    def forwardOut(self):
        # call forward before to update self.rnnLayers
        self.out = np.zeros(self.Lclass)
        self.outProb = softmax(self.rnnLayers[-1].mulV)
        self.out[np.argmax(self.outProb)] = 1
        return self.out, self.outProb
    
    def crossEntropy(self, ground):
        # call forwardOut before to update self.out
        return -np.sum(ground*np.log(self.outProb))
    
    def calcGrad(self, data, ground):
        #run after forward
        lyr = self.rnnLayers
        oldSt = np.zeros(self.Lhid)
        sDiff = np.zeros(self.Lhid)
        
        t = self.Lfeature - 1
        dmulV = self.outProb - ground
        doldS, dUt, dWt, dVt = lyr[t].backward(data[t], oldSt, self.U, self.W, self.V, sDiff, dmulV)
        oldSt = lyr[t].s
        dmulV = np.zeros(self.Lclass)
        for i in range(t-1, max(-1, t-self.bptt-1), -1): # max() is necessary for full backward propagation
            oldSi = np.zeros(self.Lhid) if i == 0 else lyr[i-1].s # full bp case
            doldS, dUi, dWi, dVi = lyr[i].backward(data[i], oldSi, self.U, self.W, self.V, doldS, dmulV)
            dUt += dUi
            dWt += dWi
        self.dUnew = self.dUnew + dUt
        self.dWnew = self.dWnew + dWt
        self.dVnew = self.dVnew + dVt # note that dVt only calculated for the last time step
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        self.dU = momentum*self.dU + self.dUnew
        self.dW = momentum*self.dW + self.dWnew
        self.dV = momentum*self.dV + self.dVnew
        self.U = self.U - learningRate*self.dU
        self.W = self.W - learningRate*self.dW
        self.V = self.V - learningRate*self.dV
        self.dUnew = np.zeros(self.U.shape)
        self.dWnew = np.zeros(self.W.shape)
        self.dVnew = np.zeros(self.V.shape)

    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target.reshape(target.shape[1]))
        self.calcGrad(sample, target.reshape(target.shape[1]))
        return loss, guess

# ==========================================
#                Q3.B
class LSTMlayer:
        # Single LSTM layer for single timestep. Stores variables,
        # and calculates gradient with stored variables.
        # forward() should be called before backward() so that
        # internal variables are set.
    def forward(self, x_t, h_prev, c_prev, params):
        self.c_prev = c_prev
        self.z = np.row_stack((h_prev,x_t))
        self.f = sigmoid(np.dot(params["Wf"],self.z)+params["bf"])
        self.i = sigmoid(np.dot(params["Wi"],self.z)+params["bi"])
        self.c_ = np.tanh(np.dot(params["Wc"],self.z)+params["bc"])
        self.c = self.f*self.c_prev + self.i*self.c_
        self.o = sigmoid(np.dot(params["Wo"],self.z)+params["bo"])
        self.h = self.o*np.tanh(self.c)
        self.v = np.dot(params["Wv"],self.h) + params["bv"]
        self.y_t = softmax(self.v)
        return self.c, self.h, self.y_t
        
    def backward(self, params, y, dh_next, dc_next):
        #run forward first for creating self parameters
        #daXXX; a denotes activaiton
        dv = self.y_t.copy() - y
        grads_step = {}
        grads_step["dWv"] = np.dot(dv, self.h.T)
        grads_step["dbv"] = dv
        dh = np.dot(params["Wv"].T, dv)
        dh += dh_next

        do = dh*np.tanh(self.c)
        da_o = do*self.o*(1-self.o)
        grads_step["dWo"] = np.dot(da_o, self.z.T)
        grads_step["dbo"] = da_o

        dc = dh*self.o*(1-np.square(np.tanh(self.c)))
        dc += dc_next

        dc_bar = dc * self.i
        da_c = dc_bar * (1-np.square(self.c))
        grads_step["dWc"] = np.dot(da_c, self.z.T)
        grads_step["dbc"] = da_c

        di = dc*self.c_
        da_i = di*self.i*(1-self.i) 
        grads_step["dWi"] = np.dot(da_i, self.z.T)
        grads_step["dbi"] = da_i

        df = dc*self.c_prev
        da_f = df*self.f*(1-self.f)
        grads_step["dWf"] = np.dot(da_f, self.z.T)
        grads_step["dbf"] = da_f

        dz = (np.dot(params["Wf"].T, da_f) + np.dot(params["Wi"].T, da_i)
             + np.dot(params["Wc"].T, da_c) + np.dot(params["Wo"].T, da_o))
    
        dh_prev = dz[:self.h.shape[0], :]
        dc_prev = self.f * dc
        return dh_prev, dc_prev, grads_step


class LSTM:
    def __init__(self, Lfeature, Lxdim, Lhid, Lclass, bptt):
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        self.Lxdim = Lxdim
        self.Lclass = Lclass
        self.bptt = bptt
        
    def initializeWeights(self):
        # Initializes the weights with Xavier and biases as one
        # Stores the variables in the dict due to the increased
        # weight and bias amount.
        rHx = np.sqrt(1/(self.Lxdim+self.Lhid))
        Lshape = (self.Lhid,self.Lxdim+self.Lhid)
        #forget gate
        self.params = {}
        self.params["Wf"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bf"] = np.ones((self.Lhid,1))
        #input gate
        self.params["Wi"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bi"] = np.ones((self.Lhid,1))
        #cell gate
        self.params["Wc"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bc"] = np.ones((self.Lhid,1))
        #output gate
        self.params["Wo"] = np.random.uniform(-rHx,rHx,Lshape)
        self.params["bo"] = np.ones((self.Lhid,1))
        #output
        self.params["Wv"] = np.random.uniform(-np.sqrt(1./self.Lhid),np.sqrt(1./self.Lhid),(self.Lclass,self.Lhid))
        self.params["bv"] = np.ones((self.Lclass,1))
        #grads
        self.grads = {}
        self.gradsNew = {}
        for key in self.params:
            self.grads["d"+key] = np.zeros(self.params[key].shape)
            self.gradsNew["d"+key] = np.zeros(self.params[key].shape)
    
    def forward(self, data):
        # data is (T,) size timeseries
        T = len(data)
        hidden, c = np.zeros((self.Lhid,1)), np.zeros((self.Lhid,1))
        foldedLayers = []
        for t in range(T):
            layer = LSTMlayer()
            hidden, c, self.outProb = layer.forward(data[t][:,None], hidden, c, self.params)
            foldedLayers.append(layer)
        self.outProb = self.outProb.T # transpose for convention
        self.Layers = foldedLayers
        return foldedLayers
    
    def forwardOut(self):
        # call forward before to update self.rnnLayers
        # out is OneHot encoded
        self.out = np.zeros(self.Lclass)
        self.out[np.argmax(self.outProb)] = 1
        return self.out, self.outProb
    
    def crossEntropy(self, ground):
        # call forward before to update self.outProb
        assert ground.shape == self.outProb.shape
        return -np.sum(ground*np.log(self.outProb))
    
    def calcGrad(self, data, ground):
        # run after forward
        # very similar to RNN. Same system is used
        lyr = self.Layers
        t = self.Lfeature - 1
        dh_next, dc_next, grads_step = lyr[t].backward(self.params, ground.T, np.zeros((self.Lhid,1)), np.zeros((self.Lhid,1)))
        for key in self.gradsNew:
            self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        for i in range(t-1, max(-1, t-self.bptt-1), -1): # change t with (self.Lfeature - 1) if you want
            y_t = lyr[i].y_t
            dh_next, dh_c, grads_step = lyr[i].backward(self.params, y_t, dh_next, dc_next) # input y_t so that dWv is zero
            for key in self.gradsNew: # add new grads to the dict for later addition
                self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        for key in self.params:
            self.grads['d'+key] = momentum*self.grads['d'+key] + self.gradsNew['d'+key]
            self.params[key] = self.params[key] - learningRate*self.grads['d'+key]
            self.gradsNew['d'+key] = np.zeros(self.params[key].shape)

    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target)
        self.calcGrad(sample, target)
        return loss, guess

# In[]
# ==========================================
#                Q3.C
class GRUlayer:
        # Single GRU layer for single timestep. Stores variables,
        # and calculates gradient with stored variables.
        # forward() should be called before backward() so that
        # internal variables are set.
    def forward(self, x_t, h_prev, params):
        # update and reset gates
        self.z = sigmoid(np.dot(params["Wz"],x_t) + np.dot(params["Uz"], h_prev) + params["bz"])
        self.r = sigmoid(np.dot(params["Wr"],x_t) + np.dot(params["Ur"], h_prev) + params["br"])
        
        # hidden units
        self.h_ = np.tanh(np.dot(params["Wh"],x_t) + np.dot(params["Uh"], np.multiply(self.r, h_prev)) + params["bh"])
        self.h = np.multiply(self.z, h_prev) + np.multiply((1-self.z), self.h_)
        
        #hid to v
        self.v = np.dot(params["Wv"],self.h) + params["bv"]
        self.y_t = softmax(self.v)
        self.h_prev = h_prev
        self.x_t = x_t
        return self.h, self.y_t
        
    def backward(self, params, y, dh_next):
        #run forward first for creating self parameters
        #daXXX; a denotes activaiton
        grads_step = {}
        
        dv = self.y_t.copy() - y
        grads_step["dWv"] = np.dot(dv, self.h.T)
        grads_step["dbv"] = dv
        
        dh = np.dot(params["Wv"].T, dv) + dh_next
        
        dh_ = np.multiply(dh, (1 - self.z))
        dh_l = dh_ * (1-np.square((self.h_))) # try tanh squared
        
        grads_step["dWh"] = np.dot(dh_l, self.x_t.T)
        grads_step["dUh"] = np.dot(dh_l, np.multiply(self.r, self.h_prev).T)
        grads_step["dbh"] = dh_l

        drh = np.dot(params["Uh"].T, dh_l)
        dr = np.multiply(drh, self.h_prev)
        dr_l = dr * self.r*(1-(self.r)) #check again

        grads_step["dWr"] = np.dot(dr_l, self.x_t.T)
        grads_step["dUr"] = np.dot(dr_l, self.h_prev.T)
        grads_step["dbr"] = dr_l
        
        dz = np.multiply(dh, self.h_prev - self.h_)
        dz_l = dz*self.z*(1-(self.z)) #check
        
        grads_step["dWz"] = np.dot(dz_l, self.x_t.T)
        grads_step["dUz"] = np.dot(dz_l, self.h_prev.T)
        grads_step["dbz"] = dz_l
        dh_prev = (np.dot(params["Uz"].T, dz_l) + np.dot(params["Ur"].T, dr_l)
                   + np.multiply(drh, self.r) + np.multiply(dh, self.z))
    
        return dh_prev, grads_step


class GRU:
    def __init__(self, Lfeature, Lxdim, Lhid, Lclass, bptt):
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        self.Lxdim = Lxdim
        self.Lclass = Lclass
        self.bptt = bptt
        
    def initializeWeights(self):
        # Weights are initialized in the same manner as LSTM
        rH = np.sqrt(1/self.Lhid)
        rX = np.sqrt(1/self.Lxdim)
        self.params = {}
        # z
        self.params["Wz"] = np.random.uniform(-rX,rX,(self.Lhid,self.Lxdim))
        self.params["Uz"] = np.random.uniform(-rH,rH,(self.Lhid, self.Lhid))
        self.params["bz"] = np.ones((self.Lhid,1))
        # r
        self.params["Wr"] = np.random.uniform(-rX,rX,(self.Lhid,self.Lxdim))
        self.params["Ur"] = np.random.uniform(-rH,rH,(self.Lhid, self.Lhid))
        self.params["br"] = np.ones((self.Lhid,1))
        # h
        self.params["Wh"] = np.random.uniform(-rX,rX,(self.Lhid,self.Lxdim))
        self.params["Uh"] = np.random.uniform(-rH,rH,(self.Lhid, self.Lhid))
        self.params["bh"] = np.ones((self.Lhid,1))
        # hid to out
        self.params["Wv"] = np.random.uniform(-rH,rH,(self.Lclass,self.Lhid))
        self.params["bv"] = np.ones((self.Lclass,1))
        # grads
        self.grads = {}
        self.gradsNew = {}
        for key in self.params:
            self.grads["d"+key] = np.zeros(self.params[key].shape)
            self.gradsNew["d"+key] = np.zeros(self.params[key].shape)
    
    def forward(self, data):
        # data is (T,) size timeseries
        T = len(data)
        hidden = np.zeros((self.Lhid,1))
        foldedLayers = []
        for t in range(T):
            layer = GRUlayer()
            hidden, self.outProb = layer.forward(data[t][:,None], hidden, self.params)
            foldedLayers.append(layer)
        self.outProb = self.outProb.T # transpose for convention
        self.Layers = foldedLayers
        return foldedLayers
    
    def forwardOut(self):
        # call forward before to update self.rnnLayers
        # out is OneHot encoded
        self.out = np.zeros(self.Lclass)
        self.out[np.argmax(self.outProb)] = 1
        return self.out, self.outProb
    
    def crossEntropy(self, ground):
        # call forward before to update self.outProb
        assert ground.shape == self.outProb.shape
        return -np.sum(ground*np.log(self.outProb))
    
    def calcGrad(self, data, ground):
        #run after forward
        lyr = self.Layers
        t = self.Lfeature - 1
        #
        dh_next, grads_step = lyr[t].backward(self.params, ground.T, np.zeros((self.Lhid,1)))
        for key in self.gradsNew:
            self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        for i in range(t-1, max(-1, t-self.bptt-1), -1): # change t with (self.Lfeature - 1) if you want
            y_t = lyr[i].y_t
            dh_next, grads_step = lyr[i].backward(self.params, y_t, dh_next) # input y_t so that dWv is zero
            for key in self.gradsNew:
                self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        for key in self.params:
            self.grads['d'+key] = momentum*self.grads['d'+key] + self.gradsNew['d'+key]
            self.params[key] = self.params[key] - learningRate*self.grads['d'+key]
            self.gradsNew['d'+key] = np.zeros(self.params[key].shape)

    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target)
        self.calcGrad(sample, target)
        return loss, guess



def mehmet_kaan_acar_21902546_mp(question):
    if question == '1' :
        print(question)
        filename = "data1.h5"

        with h5py.File(filename, "r") as f:
            groupKeys = list(f.keys())
            sets = []
            for key in groupKeys:
                sets.append(list(f[key]))
        # In[]
        images_clip = np.array(sets[0][:])
        images = images_clip.reshape((images_clip.shape[0],3,256))
        normalized = normalizeGray(images)
        # In[]
        idx = np.random.choice(images.shape[0], 200, replace=False)
        # In[]
        plotArrayGray(normalized[idx], 10, 20)
        plotArrayRGB(images[idx], 10, 20)

        # In[]
        Lhid = 8**2 #4*4, 7*7, 9*9
        lmbda = 0.0005*320
        beta = 0.005
        rho = 0.4
        Encoder = AutoEncoder(256, Lhid, lmbda, beta, rho)
        Encoder.initializeLayers(32)
        Encoder.initializeWeights()
        loss = []
        # In[]
        lr = 0.3
        mm = 0.6
        epoch = 30
        print(f"Started Training with learning rate = {lr}, momentum = {mm}, beta = {beta}, rho ={rho}")
        l, out_shuffled = trainMiniQ1(Encoder, normalized, epoch, lr, mm)
        loss.extend(l)
        # In[]
        plotArrayGray(out_shuffled[-1], 4,8)
        plotArrayGray(Encoder.out,4,8)
        plotArrayGray(Encoder.weights,8,8)
        # In[]
        plotParameterQ1(np.array(loss).T, ["Loss Function","MSE","Tykhonov","KL"], ["Loss","Auto-Encoder",Lhid,lmbda,beta,rho,lr,mm])
        # In[]
        hids = [4,7,9]
        lmbds = [0,0.0003, 0.001]
        epochs = [40,50,70]
        for i in range(3):
            for j in range(3):
        # In[]
                Lhid = hids[i]**2 #4*4, 7*7, 9*9
                lmbda = lmbds[j]
                beta = 0.005
                rho = 0.4
                Encoder = AutoEncoder(256, Lhid, lmbda, beta, rho)
                Encoder.initializeLayers(10240)
                Encoder.initializeWeights()
                loss = []
                # In[]
                lr = 0.3
                mm = 0.6
                epoch = epochs[i]
                print(f"Started Training with learning rate = {lr}, momentum = {mm}, beta = {beta}, rho ={rho}")
                l, out_shuffled = trainMiniQ1(Encoder, normalized, epoch, lr, mm)
                loss.extend(l)
                # In[]
                #plotArrayGray(Encoder.out[idx],10,20)
                plotArrayGray(Encoder.weights,hids[i],hids[i])
                # In[]
                plotParameterQ1(np.array(loss).T, ["Loss Function","MSE","Tykhonov","KL"], ["Loss","Auto-Encoder",Lhid,lmbda,beta,rho,lr,mm])

    elif question == '2' :
        print(question)
        filename = "data2.h5"

        with h5py.File(filename, "r") as f:
            groupKeys = list(f.keys())
            sets = []
            for key in groupKeys:
                sets.append(list(f[key]))
        # In[]
        testD = np.array(sets[0])
        testX = np.array(sets[1])
        trainD = np.array(sets[2][:])
        trainX = np.array(sets[3][:])
        valD = np.array(sets[4][:])
        valX = np.array(sets[5][:])
        words = [0]
        words.extend(sets[6])
        words = np.array(words, dtype=str)
        # In[]
        D, P = 8,64 #(32,256) (16,128) (8,64)
        model = fourgramNN(D, P, 250)
        model.initializeLayers(1)
        model.initializeWeights()
        lossT, lossV = [], []
        # In[]
        lr = 0.015
        mm = 0.5
        epoch = 50
        print(f"Started Training with learning rate = {lr}, momentum = {mm}")
        l1, l2 = trainMiniBatchQ2(model, trainX, trainD, valX, valD, epoch, lr, mm)
        lossV.extend(l1)
        lossT.extend(l2)
        # In[]
        idx = np.random.permutation(len(testD))
        estimateForward(model, testX[idx][:5], testD[idx][:5], words)
        # In[plot]
        plotParameterQ2([lossT, lossV], ["Training", "Validation"], ["Loss","4-Gram Model",D,P,lr,mm])
    elif question == '3' :
        # In[Read the data]
        filename = "data3.h5"

        with h5py.File(filename, "r") as f:
            groupKeys = list(f.keys())
            sets = []
            for key in groupKeys:
                sets.append(list(f[key]))
        del key

        idx = np.random.permutation(3000)
        trainX = np.array(sets[0])[idx]
        trainD = np.array(sets[1])[idx]
        testX = np.array(sets[2])
        testD = np.array(sets[3])
        valX = trainX[:300]
        valD = trainD[:300]
        trainX = trainX[300:]
        trainD = trainD[300:]

        bptt = 10
        model = RNN(150, 3, 128, 6, bptt)
        model.initializeWeights()
        lossT, lossV, accT, accTest = [], [], [], []
        # In[]
        lr = 0.001
        mm = 0.5
        epoch = 10
        print(f"Started Training with learning rate = {lr}, momentum = {mm}, bptt = {bptt}")
        l1, l2, a1, a2, confT, confTest = trainMiniBatch(model, trainX, trainD, valX, valD, testX, testD, epoch, lr, mm)
        lossT.extend(l1)
        lossV.extend(l2) 
        accT.extend(a1)
        accTest.extend(a2)
        # In[]
        plotConf(confT, "Training Set, RNN")
        plotConf(confTest, "Test Set, RNN")
        # In[plot]
        plotParameter([lossT, lossV], ["Training","Validation"], ["Loss","RNN",lr,mm,bptt])
        #%%
        plotParameter([accT, accTest], ["Training","Validation"], ["Accuracy","RNN",lr,mm,bptt])


        bptt = 3
        model = LSTM(150, 3, 128, 6, bptt)
        model.initializeWeights()
        lossT, lossV, accT, accTest = [], [], [], []
        # In[]
        lr = 0.01
        mm = 0.85
        epoch = 10
        print(f"Started Training with learning rate = {lr}, momentum = {mm}, bptt = {bptt}")
        l1, l2, a1, a2, confT, confTest = trainMiniBatch(model, trainX, trainD, valX, valD, testX, testD, epoch, lr, mm)
        lossT.extend(l1)
        lossV.extend(l2) 
        accT.extend(a1)
        accTest.extend(a2)
        # In[]
        plotConf(confT, "Training Set, RNN")
        plotConf(confTest, "Test Set, RNN")
        # In[plot]
        plotParameter([lossT, lossV], ["Training","Validation"], ["Loss","LSTM",lr,mm,bptt])
        #%%
        plotParameter([accT, accTest], ["Training","Validation"], ["Accuracy","LSTM",lr,mm,bptt])

        bptt = 10
        model = GRU(150, 3, 128, 6, bptt)
        model.initializeWeights()
        lossT, lossV, accT, accTest = [], [], [], []
        # In[]
        lr = 0.01
        mm = 0.85
        epoch = 10
        print(f"Started Training with learning rate = {lr}, momentum = {mm}, bptt = {bptt}")
        l1, l2, a1, a2, confT, confTest = trainMiniBatch(model, trainX, trainD, valX, valD, testX, testD, epoch, lr, mm)
        lossT.extend(l1)
        lossV.extend(l2) 
        accT.extend(a1)
        accTest.extend(a2)
        # In[]
        plotConf(confT, "Training Set, GRU")
        plotConf(confTest, "Test Set, GRU")
        # In[plot]
        plotParameter([lossT, lossV], ["Training","Validation"], ["Loss","GRU",lr,mm,bptt])
        #%%
        plotParameter([accT, accTest], ["Training","Validation"], ["Accuracy","GRU",lr,mm,bptt])

import sys
question = sys.argv[1]

mehmet_kaan_acar_21902546_mp(question)