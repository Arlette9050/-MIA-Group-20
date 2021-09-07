"""
Project code+scripts for 8DC00 course.
"""

import numpy as np
import matplotlib.pyplot as plt
import cad
import scipy
from IPython.display import display, clear_output
import scipy.io


def nuclei_measurement():

    fn  = '../data/nuclei_data.mat'
    mat = scipy.io.loadmat(fn)

    test_images = mat["test_images"]
    test_y      = mat["test_y"] 

    training_images = mat["training_images"] 
    training_y      = mat["training_y"]

    montage_n    = 300
    sort_ix      = np.argsort(training_y, axis=0)
    sort_ix_low  = sort_ix[:montage_n] # get the 300 smallest
    sort_ix_high = sort_ix[-montage_n:] #Get the 300 largest

    # visualize the 300 smallest and the 300 largest nuclei
    X_small = training_images[:,:,:,sort_ix_low.ravel()]
    X_large = training_images[:,:,:,sort_ix_high.ravel()]
    fig     = plt.figure(figsize=(16,8))
    ax1     = fig.add_subplot(121)
    ax2     = fig.add_subplot(122)
    cad.montageRGB(X_small, ax1)
    ax1.set_title('300 smallest nuclei')
    cad.montageRGB(X_large, ax2)
    ax2.set_title('300 largest nuclei')

    # dataset preparation
    imageSize = training_images.shape
    
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures = imageSize[0]*imageSize[1]*imageSize[2]
    training_x  = training_images.reshape(numFeatures, imageSize[3]).T.astype(float)
    test_x      = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)
    print('The size of the whole set is:', len(training_x),'samples')
    #---------------------------------------------------------------------#
    # add ones to the x training data
    training_X_ones = cad.addones(training_x)

    # find Theta using the y data and the x data
    theta, _        = cad.ls_solve(training_X_ones, training_y)

    # add ones to the x testing data
    testing_X_ones  = cad.addones(test_x)

    # find the y data found by the testing data x and the calculated theta
    predicted_y     = testing_X_ones.dot(theta)

    # compute the error between the data we have found for y and the real y data (test_y)
    E_big           = np.linalg.norm(predicted_y-test_y)**2
    E_big           = E_big/len(training_x)

       
    #---------------------------------------------------------------------#

    #training with smaller number of training samples
    #---------------------------------------------------------------------#
    # making sure we use a smaller set so we only need a part of the original training data. 
    # lets choose a smaller set by using skipping through the data. 
    # here we start at the first element (the zero position) and counting forward with ten. 
    # this means we skip 10 datapoints.
    training_X_small = training_x[::10,:]
    training_Y_small = training_y[::10,:]

    print('The size of the smaller set is:', len(training_X_small),'samples')

    # making sure we use a smaller set so we only need a part of the original training data. 
    # lets choose a smaller set by using skipping through the data. 
    # here we start at the first element (the zero position) and counting forward with ten. 
    # this means we skip 50 datapoints (even smaller dataset).
    training_X_small2 = training_x[::50,:]
    training_Y_small2 = training_y[::50,:]
    print('The size of the even smaller set is:', len(training_X_small2),'samples')
    
    # making sure we use a smaller set so we only need a part of the original training data. 
    # lets choose a smaller set by using skipping through the data. 
    # here we start at the first element (the zero position) and counting forward with ten. 
    # this means we skip 400 datapoints (very small dataset).
    training_X_small3 = training_x[::400,:]
    training_Y_small3 = training_y[::400,:]
    print('The size of a very small set is:', len(training_X_small3),'samples')

    #---------------------------------------------------------------------#

    # just as before, lets add some ones to the trainingdata
    training_X_ones_small = cad.addones(training_X_small)

    # just as before, lets add some ones to the trainingdata (even smaller dataset)
    training_X_ones_small2 = cad.addones(training_X_small2)

    # just as before, lets add some ones to the trainingdata (very small dataset)
    training_X_ones_small3 = cad.addones(training_X_small3)

    #---------------------------------------------------------------------#

    # find Theta using the y data and the x data
    thetasmall, _        = cad.ls_solve(training_X_ones_small, training_Y_small)

    # find Theta using the y data and the x data (even smaller dataset)
    thetasmall2, _        = cad.ls_solve(training_X_ones_small2, training_Y_small2)

    # find Theta using the y data and the x data (very small dataset)
    thetasmall3, _        = cad.ls_solve(training_X_ones_small3, training_Y_small3)

    #---------------------------------------------------------------------#

    # just as before, lets add some ones to the testdata
    testing_X_ones_small = cad.addones(test_x)

    #---------------------------------------------------------------------#

    # find the y data found by the testing data x and the calculated thetasmall
    predicted_y1          = testing_X_ones_small.dot(thetasmall)

    # find the y data found by the testing data x and the calculated thetasmall (even smaller dataset)
    predicted_y2          = testing_X_ones_small.dot(thetasmall2)
    
    # find the y data found by the testing data x and the calculated thetasmall (very small dataset)
    predicted_y3          = testing_X_ones_small.dot(thetasmall3)

    #---------------------------------------------------------------------#

    # compute the error between the data we have found for y and the real y data (test_y)
    E_small              = np.linalg.norm(predicted_y1-test_y)**2
    E_small              = E_small/len(training_X_small)

    # compute the error between the data we have found for y and the real y data (test_y)(even smaller dataset)
    E_small2              = np.linalg.norm(predicted_y2-test_y)**2
    E_small2              = E_small2/len(training_X_small2)

    # compute the error between the data we have found for y and the real y data (test_y)(very small dataset)
    E_small3              = np.linalg.norm(predicted_y3-test_y)**2
    E_small3              = E_small3/len(training_X_small3)

    #---------------------------------------------------------------------#

    # print the error data
    print('The mean squared error for the whole set of training samples for the linear regression model is:',E_big)
    # print the error data
    print('The mean squared error for a smaller number of training samples for the linear regression model is:',E_small)
    # print the error data (even smaller dataset)
    print('The mean squared error for an even smaller number of training samples for the linear regression model is:',E_small2)
    # print the error data (very small dataset)
    print('The mean squared error for very small number of training samples for the linear regression model is:',E_small3)
    #---------------------------------------------------------------------#

    # visualize the results
    
    fig2   = plt.figure(figsize=(16,8))
    ax1    = fig2.add_subplot(221)
    line1, = ax1.plot(predicted_y, test_y, ".g", markersize=3)
    ax1.grid()
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Predicted Area')
    ax1.set_title('Training with full sample')

    ax2    = fig2.add_subplot(222)
    line2, = ax2.plot(predicted_y1, test_y, ".g", markersize=3)
    ax2.grid()
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Predicted Area')
    ax2.set_title('Training with smaller sample')

    ax3    = fig2.add_subplot(223)
    line3, = ax3.plot(predicted_y2, test_y, ".g", markersize=3)
    ax3.grid()
    ax3.set_xlabel('Area')
    ax3.set_ylabel('Predicted Area')
    ax3.set_title('Training with an even smaller sample')

    ax4    = fig2.add_subplot(224)
    line4, = ax4.plot(predicted_y3, test_y, ".g", markersize=3)
    ax4.grid()
    ax4.set_xlabel('Area')
    ax4.set_ylabel('Predicted Area')
    ax4.set_title('Training with a very small sample')


def nuclei_classification():
    ## dataset preparation
    fn  = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images       = mat["test_images"] # (24, 24, 3, 20730)
    test_y            = mat["test_y"] # (20730, 1)
    training_images   = mat["training_images"] # (24, 24, 3, 14607)
    training_y        = mat["training_y"] # (14607, 1)
    validation_images = mat["validation_images"] # (24, 24, 3, 14607)
    validation_y      = mat["validation_y"] # (14607, 1)

    #for a smaller dataset of 0.5% of the original set, otherwise comment out next line
    #training_y        = training_y[::200,:]
    
    
    ## dataset preparation
    imageSize    = training_images.shape
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures  = imageSize[0]*imageSize[1]*imageSize[2]
    training_x   = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
    validation_x = validation_images.reshape(numFeatures, validation_images.shape[3]).T.astype(float)
    test_x       = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    # for a smaller dataset of 0.5% of the original set, otherwise comment out next line
    #training_x        = training_x[::200,:] 

    # the training will progress much better if we
    # normalize the features
    meanTrain    = np.mean(training_x, axis=0).reshape(1,-1)
    stdTrain     = np.std(training_x, axis=0).reshape(1,-1)

    training_x   = training_x - np.tile(meanTrain, (training_x.shape[0], 1))
    training_x   = training_x / np.tile(stdTrain, (training_x.shape[0], 1))

    validation_x = validation_x - np.tile(meanTrain, (validation_x.shape[0], 1))
    validation_x = validation_x / np.tile(stdTrain, (validation_x.shape[0], 1))

    test_x       = test_x - np.tile(meanTrain, (test_x.shape[0], 1))
    test_x       = test_x / np.tile(stdTrain, (test_x.shape[0], 1))

    ## training linear regression model
    #-------------------------------------------------------------------#
    # setting the learning rate
    mu         = 0.0005

    #setting the batch size
    batch_size = 100

    # setting the number of iterations
    num_iterations = 1000

    # setting Theta (model parameter)
    Theta = np.random.normal(0,0.01,(numFeatures+1, 1))

    # stop, if false do not use. if true use it as a critetia later on to stop the iterations.
    stop = True

    #hyperparameter for the momentum
    Zeta = 0.0001

    #-------------------------------------------------------------------#
    xx      = np.arange(num_iterations)
    loss    = np.empty(*xx.shape)
    loss[:] = np.nan
    validation_loss    = np.empty(*xx.shape)
    validation_loss[:] = np.nan
    g    = np.empty(*xx.shape)
    g[:] = np.nan

    fig = plt.figure(figsize=(8,8))
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (average per sample)')
    ax2.set_title('Initial mu = '+str(mu))
    h1, = ax2.plot(xx, loss, linewidth=2) #'Color', [0.0 0.2 0.6],
    h2, = ax2.plot(xx, validation_loss, linewidth=2) #'Color', [0.8 0.2 0.8],
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0, num_iterations)
    ax2.grid()

    text_str2 = 'iter.: {}, loss: {:.3f}, val. loss: {:.3f}'.format(0, 0, 0)
    txt2      = ax2.text(0.3, 0.95, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)
    # use the line mentioned below when using only 0.5% of the data and comment out the line above:
    #txt2      = ax2.text(0.5, 0.1, text_str2, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10}, transform=ax2.transAxes)

    # lets make sure the graph stays steady in its difference, then we know we have the right solution
    patience = 0
    # create a variable for the prior iteration which for the first iteration is zero and will be updated if the new loss function and this one will not differ significantly.
    prioriter = 0
  
    #add momentum
    Momentum = 0

    #counter make
    counter = 1

    
    
    for k in np.arange(num_iterations):
        # pick a batch at random
        idx = np.random.randint(training_x.shape[0], size=batch_size)

        training_x_ones   = cad.addones(training_x[idx,:])
        validation_x_ones = cad.addones(validation_x)

        # the loss function for this particular batch
        loss_fun = lambda Theta: cad.lr_nll(training_x_ones, training_y[idx], Theta)

        # gradient descent
        # instead of the numerical gradient, we compute the gradient with
        # the analytical expression, which is much faster
        Momentum    = Momentum + cad.lr_agrad(training_x_ones, training_y[idx], Theta).T
        # mean gradient
        Momentumnew = Zeta*(Momentum/counter)
        
        Theta_new   = Theta - mu*cad.lr_agrad(training_x_ones, training_y[idx], Theta).T
        Theta_new   = Theta_new-Momentumnew
       
        
        
        loss[k]            = loss_fun(Theta_new)/batch_size
        validation_loss[k] = cad.lr_nll(validation_x_ones, validation_y, Theta_new)/validation_x.shape[0]

        counter = counter + 1
        

        # visualize the training
        h1.set_ydata(loss)
        h2.set_ydata(validation_loss)
        text_str2 = 'iter.: {}, loss: {:.3f}, val. loss={:.3f}, p. loss = {:.3f}'.format(k, loss[k], validation_loss[k], prioriter)
        txt2.set_text(text_str2)

       

        Theta     = None
        Theta     = np.array(Theta_new)
        Theta_new = None
        tmp       = None

        # now to introduce the stop. here we need to look at the change of the loss function of the current iteration and the one before that. 
        # when the lossfunction of the prior iteration is signigficantly bigger than the one before the graph should keep on rendering otherwise (when no significant difference is found anymore) it should go on. 
        # if significantly bigger the graph should go on to the next iterations and the mu should be updated.
        # the error in the nuclei_measurement goes into the 1e-9 so a real difference can be found when looking at 1e-9.
        # when the lossfunction of the prior iteration is bigger than validation_loss, a negative number will be found (this will always be smaller than 1e-9).
        # to solve this problem, the absolute value for the difference has to be taken.

        # enter a patience, to make it a steady difference which doesn't move

        if abs(validation_loss[k]-prioriter)<1e-4 and stop:
            patience = patience+1
        else:
            patience = 0
         # if the difference in loss doesn't change significantly for ten iterations in a row the graph should stop, this means a steady graph is found
        if patience == 10: 
            break    
        
        # for the first iterations prioriter is 0. now we update the prioriter, because by now the abs(validation_loss[k]-prioriter) is bigger than 1e-9 (otherwise it whould have stopped).
        # so now the current iterations has to become the last iteration, as we will begin with a new iteration after the mu is updated.
        prioriter = validation_loss[k]

        # update mu
        if (patience>0) :
            mu = mu
        else:
            mu = mu*(1-k/num_iterations)
        
        print('The last mu is:',mu)


        display(fig)
        clear_output(wait = True)
        plt.pause(.005)
    #Comment out next two lines when using 0.5% of the dataset!
    else: 
        print('Maximum iterations reached')

    writing = []
    for i in range(len(Theta)):
        th = Theta[i,:]
        writing.append(th)
    with open('Thetas.txt','w+') as file:
        file.write(str(writing))




def loadmodel():
    with open('Thetas.txt') as file:
        Thetas = file.read()
    Thetas = Thetas.split(',')
    
    Thetas = [th.strip(' [array(])') for th in Thetas]
    Thetas = [float(th) for th in Thetas]

    cutoff = 0.5

    ## dataset preparation
    fn  = '../data/nuclei_data_classification.mat'
    mat = scipy.io.loadmat(fn)

    test_images       = mat["test_images"] # (24, 24, 3, 20730)
    test_y            = mat["test_y"] # (20730, 1)
    training_images   = mat["training_images"] # (24, 24, 3, 14607)
    training_y        = mat["training_y"] # (14607, 1)
    validation_images = mat["validation_images"] # (24, 24, 3, 14607)
    validation_y      = mat["validation_y"] # (14607, 1)
    
    ## dataset preparation
    imageSize    = training_images.shape
    # every pixel is a feature so the number of features is:
    # height x width x color channels
    numFeatures  = imageSize[0]*imageSize[1]*imageSize[2]
    training_x   = training_images.reshape(numFeatures, training_images.shape[3]).T.astype(float)
    test_x       = test_images.reshape(numFeatures, test_images.shape[3]).T.astype(float)

    # the training will progress much better if we
    # normalize the features
    meanTrain    = np.mean(training_x, axis=0).reshape(1,-1)
    stdTrain     = np.std(training_x, axis=0).reshape(1,-1)

    test_x       = test_x - np.tile(meanTrain, (test_x.shape[0], 1))
    test_x       = test_x / np.tile(stdTrain, (test_x.shape[0], 1))

    TEST_X_ONES = cad.addones(test_x)
    predictedy = cad.sigmoid(TEST_X_ONES.dot(Thetas))
    predictedy = np.round(predictedy)

    TruepositiveandTruenegative = np.diag(predictedy==test_y)
    TPTN = np.sum(TruepositiveandTruenegative)
    all = len(predictedy)

    Accuracy = TPTN/all

    print('The accuracy for this predicted model is:', Accuracy)

    Falsepositives = np.sum(np.diag(predictedy[predictedy==1] != test_y[predictedy==1]))
    Falsenegatives = np.sum(np.diag(predictedy[predictedy==0] != test_y[predictedy==0]))
    Truepositives  = np.sum(np.diag(predictedy[predictedy==1] == test_y[predictedy==1]))
    Truenegatives  = np.sum(np.diag(predictedy[predictedy==0] == test_y[predictedy==0]))

    Sensitivity = Truepositives/(Truepositives+Falsenegatives)
    Specificity = Truenegatives/(Truenegatives+Falsepositives)

    print('The sensitivity of this model is:',Sensitivity)
    print('The specificity of this model is:',Specificity)

    



