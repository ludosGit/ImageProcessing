import torch
import losses
import numpy as np
from tqdm import tqdm  # progressbar decorator for iterators
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics



########## TRAIN A CONTRASTIVE DATASET USING THE CONTRASTIVE LOSS FUNCTION ##############
# INPUT
# train_dl: training data loader
# train_data: torch dataset with the correct attributes as defined in 'dataset.py'
# model: network architecture
# epochs: number of epochs
# epochs_toplot: list of the index of the epochs at the end of which I want to plot the train embeddings
# opt: optimizer used
# labels: list of the UNIQUE labels; the data are classified according to them
# scheduler is an LR scheduler object from torch.optim.lr_scheduler (update learning rate)
# unsq tells if we have to unsqueeze the images (adding one dimension) before passing them through the model (helpful because The pretrained networks take in input one more dimension)


def fit_contrastive(train_dl, train_data, model, epochs, opt, labels, epochs_toplot=None, unsq=False, scheduler=None, device = 'cpu', margin = 1):
    '''
    epochs_toplot: list telling after which epochs plot the data in the plane
    NOTE: plot only if the final dimension is 2!
    margin: margin in the definition of contrastive loss
    '''
    loss_per_epoch_vector = []
    loss_func = losses.ContrastiveLoss(margin)
    for epoch in range(epochs):
        train_data.set_train(True)
        running_loss = []
        model.train()
        for _, (anchor_img, paired_img, dummy) in enumerate(tqdm(train_dl, desc="Training", leave=False)):
            # POSSIBLE PROBLEM: THE PAIRED IMAGE IS NOT TAKEN IN THE BATCH BUT IN THE WHOLE DATASET
            anchor_img = anchor_img.to(device)
            paired_img = paired_img.to(device)
            F1 = model(anchor_img).to(device)
            F2 = model(paired_img).to(device)
            dummy = dummy.to(device)
            loss = loss_func(F1, F2, dummy)
            loss.backward() # computes the gradients w.r.t. the observations in the batch
            running_loss.append(loss.cpu().detach().numpy())  # simply convert a torch.tensor into a numpy array
            opt.step() # method to update the parameters using the criterion given by opt (ex SGD)
            opt.zero_grad() # sets the gradients to zero
        if scheduler:
            scheduler.step()

        epoch_loss = np.mean(running_loss)
        loss_per_epoch_vector.append(epoch_loss)
        #  print the mean error per epoch
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

        # PLOT THE IMAGES IN THE EMBEDDING SPACE FOR SOME EPOCHS
        if epochs_toplot is not None and epoch in epochs_toplot:
            train_data.set_train(False)
            train_results = []
            model.eval()
            # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
            with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
                for i in train_data.index:
                    image = train_data[i] # .unsqueeze(0)  add batch dimension to be readable by conv2D
                    if unsq:
                        image = image.unsqueeze(0)
                    train_results.append(model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

            train_results = np.concatenate(train_results)  # Join a sequence of arrays along an existing axis
            y_train = train_data.labels
            plt.figure(figsize=(15, 10), facecolor="azure")
            plt.title('Embedding space EPOCH {}'.format(epoch+1))
            for label in labels:  # labels numpy array of the unique labels
                coord = train_results[y_train == label]  # select the indexes for images with the same label
                plt.scatter(coord[:, 0], coord[:, 1],
                            label=label)  # scatter plot of the different classes in the embedding space
            plt.legend()
            plt.show()
        
    train_data.set_train(False)
    train_results = []
    model.eval()
    # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
    with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
        for i in train_data.index:
            image = train_data[i] # .unsqueeze(0)  add batch dimension to be readable by conv2D
            if unsq:
                image = image.unsqueeze(0)
            train_results.append(model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

    train_results = np.concatenate(train_results)  # Join a sequence of arrays along an existing axis
    print('Shape of the data in the learned metric space:', train_results.shape)
    return loss_per_epoch_vector, train_results # return train_results of the last iteration, I will need them to apply k-nn to test

########## TRAIN A TRIPLET DATASET USING THE TRIPLET LOSS FUNCTION##############

def fit_triplet(train_dl, train_data, model, epochs, opt, labels, unsq=False, scheduler=None, device = 'cpu', margin = 1, epochs_toplot=None):
    loss_per_epoch_vector = []
    loss_func = losses.TripletLoss(margin)
    for epoch in range(epochs):
        train_data.set_train(True)
        running_loss = []
        model.train()
        for _, (anchor_img, positive_img, negative_img) in enumerate(tqdm(train_dl, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            F_anchor = model(anchor_img).to(device)
            F_positive = model(positive_img).to(device)
            F_negative = model(negative_img).to(device)
            loss = loss_func(F_anchor, F_positive, F_negative, margin)
            loss.backward() # computes the gradients w.r.t. the observations in the batch
            running_loss.append(loss.cpu().detach().numpy())  # simply convert a torch.tensor into a numpy array
            opt.step() # method to update the parameters using the criterion given by opt (ex SGD)
            opt.zero_grad() # sets the gradients to zero
        if scheduler:
            scheduler.step()
        epoch_loss = np.mean(running_loss)
        loss_per_epoch_vector.append(epoch_loss)
        # print the mean error per epoch
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

        if epochs_toplot is not None and epoch in epochs_toplot:
            train_data.set_train(False)
            train_results = []
            model.eval()
            # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
            with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
                for i in train_data.index:
                    image = train_data[i] # .unsqueeze(0)  add batch dimension to be readable by conv2D
                    if unsq:
                        image = image.unsqueeze(0)
                    train_results.append(model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

            train_results = np.concatenate(train_results)  # Join a sequence of arrays along an existing axis

            y_train = train_data.labels
            plt.figure(figsize=(15, 10), facecolor="azure")
            plt.title('Embedding space EPOCH {}'.format(epoch+1))
            for label in labels:  # labels numpy array of the unique labels
                coord = train_results[y_train == label]  # select the indexes for images with the same label
                plt.scatter(coord[:, 0], coord[:, 1],
                            label=label)  # scatter plot of the different classes in the embedding space
            plt.legend()
            plt.show()
        train_data.set_train(False)
        train_results = []
        model.eval()
        # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
        with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
            for i in train_data.index:
                image = train_data[i] # .unsqueeze(0)  add batch dimension to be readable by conv2D
                if unsq:
                    image = image.unsqueeze(0)
                train_results.append(model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

        train_results = np.concatenate(train_results)  # Join a sequence of arrays along an existing axis
    print('Shape of the data in the learned metric space:', train_results.shape)

    return loss_per_epoch_vector, train_results


#### INTRODUCE A VALIDATION SET ######

def fit_contrastive_validation(train_dl, train_data, valid_data, model, epochs, epochs_toplot, opt, labels, unsq=False, scheduler=None, device = 'cpu', margin = 1):
    # epochs to plot: list containing the epochs indexes I want to plot. MUST contain the last epoch in order to return the last training_results
    loss_per_epoch_vector = []
    loss_func = losses.contrastive_loss
    for epoch in range(epochs):
        train_data.set_train(True)
        running_loss = []
        model.train()
        for step, (anchor_img, paired_img, dummy) in enumerate(tqdm(train_dl, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            paired_img = paired_img.to(device)
            F1 = model(anchor_img).to(device)
            F2 = model(paired_img).to(device)
            dummy = dummy.to(device)
            loss = loss_func(F1, F2, dummy, margin)
            loss.backward()  # computes the gradients w.r.t. the observations in the batch
            running_loss.append(loss.cpu().detach().numpy())  # simply convert a torch.tensor into a numpy array
            opt.step()  # method to update the parameters using the criterion given by opt (ex SGD)
            opt.zero_grad()  # sets the gradients to zero
        if scheduler:
            scheduler.step()
        epoch_loss = np.mean(running_loss)
        loss_per_epoch_vector.append(epoch_loss)
        # print the mean error per epoch
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

        if epoch in epochs_toplot:
            train_data.set_train(False)
            train_results = []
            model.eval()
            # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
            with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
                for i in train_data.index:
                    image = train_data[i] # .unsqueeze(0)  add batch dimension to be readable by conv2D
                    if unsq:
                        image = image.unsqueeze(0)
                    train_results.append(model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

            train_results = np.concatenate(train_results)  # Join a sequence of arrays along an existing axis
            print(train_results.shape)  # 2 coordinates for all the 50000 observations

            y_train = train_data.labels
            plt.figure(figsize=(15, 10), facecolor="azure")
            plt.title('Embedding space EPOCH {}'.format(epoch+1))
            for label in labels:  # labels numpy array of the unique labels
                coord = train_results[y_train == label]  # select the indexes for images with the same label
                plt.scatter(coord[:, 0], coord[:, 1],
                            label=label)  # scatter plot of the different classes in the embedding space
            plt.legend()
            plt.show()

    # at the end of the epoch loop evaluation on validation data:
    valid_data.set_train(False)
    valid_results = []
    model.eval()
    # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
    with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
        for i in valid_data.index:
            image = valid_data[i]
            valid_results.append(
                model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

    valid_results = np.concatenate(valid_results)  # Join a sequence of arrays along an existing axis
    print('The shape of the validation results is', valid_results.shape)  # 2 coordinates for all the 10000 observations
    neigh = KNeighborsClassifier(n_neighbors=10)
    y_train = train_data.labels
    neigh.fit(train_results, y_train)
    y_pred = neigh.predict(valid_results)
    y_valid = valid_data.labels
    valid_accuracy = metrics.accuracy_score(y_valid, y_pred)

    return loss_per_epoch_vector, valid_accuracy


def fit_triplet_validation(train_dl, train_data, valid_data, model, epochs, epochs_toplot, opt, labels, unsq=False, scheduler=None, device = 'cpu', margin = 1):
    # epochs to plot: list containing the epochs indexes I want to plot. MUST contain the last epoch in order to return the last training_results
    loss_per_epoch_vector = []
    loss_func = losses.triplet_loss
    for epoch in range(epochs):
        train_data.set_train(True)
        running_loss = []
        model.train()
        for step, (anchor_img, positive_img, negative_img) in enumerate(tqdm(train_dl, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            F_anchor = model(anchor_img).to(device)
            F_positive = model(positive_img).to(device)
            F_negative = model(negative_img).to(device)
            loss = loss_func(F_anchor, F_positive, F_negative, margin)
            loss.backward() # computes the gradients w.r.t. the observations in the batch
            running_loss.append(loss.cpu().detach().numpy())  # simply convert a torch.tensor into a numpy array
            opt.step() # method to update the parameters using the criterion given by opt (ex SGD)
            opt.zero_grad() # sets the gradients to zero
        epoch_loss = np.mean(running_loss)
        loss_per_epoch_vector.append(epoch_loss)

        # print the mean error per epoch
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

        if epoch in epochs_toplot:
            train_data.set_train(False)
            train_results = []
            model.eval()
            # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
            with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
                for i in train_data.index:
                    image = train_data[i] # .unsqueeze(0)  add batch dimension to be readable by conv2D
                    if unsq:
                        image = image.unsqueeze(0)
                    train_results.append(model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

            train_results = np.concatenate(train_results)  # Join a sequence of arrays along an existing axis
            print(train_results.shape)  # 2 coordinates for all the 50000 observations

            y_train = train_data.labels
            plt.figure(figsize=(15, 10), facecolor="azure")
            plt.title('Embedding space EPOCH {}'.format(epoch+1))
            for label in labels:  # labels numpy array of the unique labels
                coord = train_results[y_train == label]  # select the indexes for images with the same label
                plt.scatter(coord[:, 0], coord[:, 1],
                            label=label)  # scatter plot of the different classes in the embedding space
            plt.legend()
            plt.show()

    # at the end of the epoch loop evaluation on validation data:
    valid_data.set_train(False)
    valid_results = []
    model.eval()
    # # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
    with torch.no_grad():  # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
        for i in valid_data.index:
            image = valid_data[i]
            valid_results.append(
                model(image.to(device)).cpu().numpy())  # convert the output of the network into numpy array

    valid_results = np.concatenate(valid_results)  # Join a sequence of arrays along an existing axis
    print('The shape of the validation results is', valid_results.shape)  # 2 coordinates for all the 10000 observations
    neigh = KNeighborsClassifier(n_neighbors=10)
    y_train=train_data.labels
    neigh.fit(train_results, y_train)
    y_pred = neigh.predict(valid_results)
    y_valid = valid_data.labels
    valid_accuracy = metrics.accuracy_score(y_valid, y_pred)

    return loss_per_epoch_vector, valid_accuracy


######### ATTEMPT TO DEFINE A TRAINING FOR RANKED LIST LOSS ######

# !!!! NOTE: in input I need train data to have access to the main attributes
# NeedED to perform a non trivial sample mining for each anchor image

# INPUT: T, alpha are hyperparameters of the loss function

def fit_rankedlist(train_dl, train_data, model, epochs, opt, T, alpha, margin = 1, device = 'cpu'):

    model.train()
    indexes = train_data.index
    labels = train_data.labels
    pairwise_loss = losses.pairwise_margin_loss
    images = train_data.images
    for epoch in range(epochs):
        running_loss = []
        # Non-trivial Sample Mining will be in the training loop
        for step, (batch, batch_indexes) in enumerate(tqdm(train_dl, desc="Training", leave=False)):
            print(batch_indexes.shape)
            # the dimension is batch_size
            total_loss = torch.tensor([])
            for anchor_img, index in zip(batch, batch_indexes):
                anchor_img = anchor_img.to(device)
                anchor_label = labels[index]
                batch_labels = labels[batch_indexes]
                positive_list = batch_indexes[batch_labels == anchor_label]
                negative_list = batch_indexes[batch_labels != anchor_label]
                positive_images = images[positive_list]
                negative_images = images[negative_list]
                l_p = torch.tensor([])  # list of the relevant positive losses
                l_n = torch.tensor([])   # negative losses
                weights = torch.tensor([])
                F = model(anchor_img)
                for positive_im in positive_images:  # mining non trivial positive examples
                    F_positive = model(positive_im)  # calculate positive embeddings
                    loss = pairwise_loss(F, F_positive, 1, alpha, margin)
                    if loss > 0:
                        l_p = torch.cat((l_p, torch.tensor([loss])),0)

                for negative_im in negative_images:
                    F_negative = model(negative_im)  # calculate negative embeddings
                    loss = pairwise_loss(F, F_negative, 0, alpha, margin)
                    if loss > 0:
                        l_n = torch.cat((l_n, torch.tensor([loss])),0)
                        weight = math.exp(T * loss)
                        weights = torch.cat((weights, torch.tensor([weight])),0)
            l_p = torch.mean(l_p)
            weights = weights / torch.sum(weights)
            l_n = torch.sum(l_n * weights)
            total_loss = l_p + l_n
            total_loss.backward() # computes the gradients w.r.t. the observations in the batch
            running_loss.append(loss.cpu().detach().numpy())  # simply convert a torch.tensor into a numpy array
            opt.step() # method to update the parameters using the criterion given by opt (ex SGD)
            opt.zero_grad() # sets the gradients to zero
        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
    # print the mean error per epoch
    return None



