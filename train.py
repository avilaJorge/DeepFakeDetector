import torch
import time
from datetime import datetime
from models import save_model
from utils import ProgressMonitor, RunningAverage, check_dims


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_loss    = float('inf')
prev_loss    = float('inf')
loss_inc_cnt = 0
stp_erly_cnt = 1
stop_early   = False
dt = datetime.now().strftime("%m_%d_%H_%M")

def train(model,
                optimizer,
                criterion,
                dataloader, 
                validation_loader,
                model_name,
                path,
                predicter=None,
                dims_checker=check_dims,
                s_epoch=1, 
                num_epochs=100):
    
    global best_loss
    global prev_loss
    global loss_inc_cnt
    global stop_early
    global dt

    train_losses = []
    valid_losses = []

    dt = datetime.now().strftime("%m_%d_%H_%M")
    # Train the models
    total_step = len(dataloader.dataset)
    for epoch in range(s_epoch, num_epochs):
        
        # create a progress bar
        progress = ProgressMonitor(length=total_step)
                                   
        train_loss = RunningAverage()
        
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            # Move to GPU
            x = dims_checker(x).to(device)
            y = y.to(device)
            # Forward, backward and optimize

            pred = model(x)

            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            loss=loss.item()
                                   
            # update average loss
            train_loss.update(loss)

            # update progress bar
            progress.update(x.shape[0], train_loss)


        print('Epoch: ', epoch)
        print('Training loss:', train_loss)
        evaluate(model, 
                optimizer,
                criterion,
                epoch, 
                dataloader, 
                train_losses,
                model_name,
                path,
                predicter=predicter,
                dims_checker=dims_checker,
                validation=False, 
                name="Training")
        evaluate(model, 
                optimizer, 
                criterion,
                epoch, 
                validation_loader, 
                valid_losses,
                model_name,
                path,
                predicter=predicter,
                dims_checker=dims_checker)

        if stop_early:
            break
    
    # Reset Global variables
    best_loss    = float('inf')
    prev_loss    = float('inf')
    loss_inc_cnt = 0
    stop_early   = False
    return train_losses, valid_losses

def evaluate(model,
            optimizer,
            criterion,
            epoch, 
            data_loader, 
            valid_losses,
            model_name,
            path,
            dims_checker=check_dims,
            predicter=None,
            validation=True, 
            name="Validation"):

    global best_loss
    global prev_loss
    global loss_inc_cnt
    global stop_early
    global dt    

    best_model_fn = path + "best_model_" + dt + ".pt"
    model_fn      = path + "model_" + dt + ".pt"

    # keep track of predictions
    y_predics = []
    y_targets = []
    predics   = []
    
    with torch.no_grad():
                            
        total_step = len(data_loader.dataset)
        
        # create a progress bar
        progress = ProgressMonitor(length=total_step)
        
        losses = RunningAverage()
        
        for i, (x, y) in enumerate(data_loader):

            # Move to GPU
            x = dims_checker(x).to(device)
            y = y.to(device)

            # Evaluate
            pred = model(x)
            
            loss = criterion(pred, y).item()
                            
            losses.update(loss)

            if predicter is not None: 
                pred = predicter(pred)
            predics.extend(pred)
            y_predics.extend((pred > 0.5).float())
            y_targets.extend(y)
            
            # update progress bar
            progress.update(x.shape[0], losses)
                            
                                   
        
        loss = losses.value
        print(name + " Loss: ", losses)
        valid_losses.append(losses.value)

        # Calculate validation accuracy
        y_pred = torch.tensor(y_predics, dtype=torch.int64)
        y_targ = torch.tensor(y_targets, dtype=torch.int64)
        accuracy = torch.mean((y_pred == y_targ).float())
        print(name + ' accuracy: {:.4f}%'.format(float(accuracy) * 100))

        if validation:
            if best_loss > loss:
                best_loss = loss
                print('"Best Loss": ' + str(best_loss) + '\n')
                save_model(model, 'Best-' + model_name, dt, path)

            loss_inc_cnt = loss_inc_cnt + 1 if prev_loss < loss else 0
            if loss_inc_cnt > stp_erly_cnt: stop_early = True
            save_model(model, model_name, dt, path)
            print("Validation Loss has gone up %d times.\n" % (loss_inc_cnt))
            prev_loss = loss
            if stop_early: 
                return loss
        
        return loss
