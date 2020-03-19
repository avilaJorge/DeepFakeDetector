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
predictions  = []
dt = datetime.now().strftime("%m_%d_%H_%M")

def train(model,
                optimizer,
                criterion,
                dataloader, 
                validation_loader,
                model_name,
                path,
                acc_thresh=0.5,
                predicter=None,
                label_transform=None,
                dims_checker=check_dims,
                s_epoch=1, 
                num_epochs=100,
                choice = None):
    
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
        
        for i, (x, y, _, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # change the filter
            n = x.shape[1]
            part_freq = int(n/3)
            if choice == 'low_pass':
                x = x[:, :part_freq]
            elif choice == 'band_pass':
                x = x[:, part_freq:2*part_freq]
            elif choice == 'high_pass':
                x = x[:, n-part_freq:]
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
                acc_thresh=acc_thresh,
                label_transform=label_transform,
                predicter=predicter,
                dims_checker=dims_checker,
                validation=False, 
                name="Training",
                choice = choice)
        evaluate(model, 
                optimizer, 
                criterion,
                epoch, 
                validation_loader, 
                valid_losses,
                model_name,
                path,
                acc_thresh=acc_thresh,
                label_transform=label_transform,
                predicter=predicter,
                dims_checker=dims_checker,
                choice = choice)

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
            acc_thresh=0.5,
            dims_checker=check_dims,
            eval_info=None,
            predicter=None,
            label_transform=None,
            validation=True, 
            name="Validation",
            choice = None):

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
    img_paths = []
    img_class = []
    img_data  = []
    
    with torch.no_grad():
                            
        total_step = len(data_loader.dataset)
        
        # create a progress bar
        progress = ProgressMonitor(length=total_step)
        
        losses = RunningAverage()
        
        for i, (x, y, img_path, img_c) in enumerate(data_loader):
            
            # change the filter
            n = x.shape[1]
            part_freq = int(n/3)
            if choice == 'low_pass':
                x = x[:, :part_freq]
            elif choice == 'band_pass':
                x = x[:, part_freq:2*part_freq]
            elif choice == 'high_pass':
                x = x[:, n-part_freq:]

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
            y_predics.extend((pred > acc_thresh).float())
            y_targets.extend(y if label_transform is None else label_transform(y))
            
            if eval_info is not None:
                img_paths.extend(img_path)
                img_class.extend(img_c)
                img_data.extend(x)
            
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
        
        if eval_info is not None:
            eval_info['y_predics'] = y_predics
            eval_info['y_targets'] = y_targets
            eval_info['predics']   = predics
            eval_info['img_paths'] = img_paths
            eval_info['img_class'] = img_class
            eval_info['img_data']  = img_data
            eval_info['accuracy']  = (y_pred == y_targ).int()

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
