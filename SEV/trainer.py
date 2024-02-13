from tqdm import tqdm

# this is the implementation of the training process      

def model_train(model, original_criterion, criterion, optimizer, train_loader, num_epochs, warm_up):
    # loop through num_epochs
    for epoch in tqdm(range(num_epochs)):
        for x, y in train_loader:
            outputs = model(x).squeeze()
            # if the epochs are less than warm_up, use the BCE loss
            if epoch < num_epochs * warm_up:
                original_loss = original_criterion(outputs, y)
                optimizer.zero_grad()
                original_loss.backward()
            # if the epochs are greater than warm_up, use the SEV loss
            else:
                _, loss, _= criterion(outputs, y, x)
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()

        
        