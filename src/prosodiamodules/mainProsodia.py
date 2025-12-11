import os
import sys
import errno
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from prosodiamodules.prosodiaDataset import ProsodiaDataset
from prosodiaDataset import load_dataset
from prosodiaDataset import pad
from prosodiamodules.modelBertProsodia import Bert
"""
Archivo obtenido de Helsinki-NLP/prosody (MIT License 2019)
https://github.com/Helsinki-NLP/prosody
"""
def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def weighted_mse_loss(input,target):
    tgt_device = target.device
    BUFFER = torch.Tensor([3.0]).to(tgt_device)
    SOFT_MAX_BOUND = torch.Tensor([6.0]).to(tgt_device) + BUFFER
    weights = (torch.min(target + BUFFER, SOFT_MAX_BOUND) / SOFT_MAX_BOUND)
    weights = weights / torch.sum(weights)
    weights = weights.cuda()
    sq_err = (input-target)**2
    weighted_err = sq_err * weights.expand_as(target)
    loss = weighted_err.mean()
    return loss

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def main():

    config_dict = {
        "datadir": ".",
        "train_set": "train_100",
        "batch_size": 32,
        "epochs": 2,
        "model": "BertUncased",
        "nclasses": 3,
        "hidden_dim": 600,
        "embedding_file": "embeddings/glove.840B.300d.txt",
        "layers": 1,
        "save_path": "results.txt",
        "log_every": 10,
        "learning_rate": 0.00005,
        "weight_decay": 0,
        "fraction_of_train_data": 1,
        "optimizer": "adam",
        "ignore_punctuation": True,
        "sorted_batches": False,
        "mask_invalid_grads": False,
        "invalid_set_to": -2.0,
        "log_values": False,
        "weighted_mse": False,
        "shuffle_sentences": False,
        "seed": 1234,
        "save_every_iterations": 100, # New parameter
    }

    config = Config(config_dict)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU (torch.device({})).".format(device))
    else:
        device = torch.device('cpu')
        print("GPU not available so training on CPU (torch.device({})).".format(device))


    optim_algorithm = optim.Adam
    dataset = ProsodiaDataset()
    splits, tag_to_index, index_to_tag, vocab = load_dataset(config)

    model = Bert(device, labels=len(tag_to_index))
    model.to(device)

    train_dataset = ProsodiaDataset(splits["train"], tag_to_index, config)
    eval_dataset = ProsodiaDataset(splits["dev"], tag_to_index, config)
    test_dataset = ProsodiaDataset(splits["test"], tag_to_index, config)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=not(config.sorted_batches),
                                 num_workers=1,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=eval_dataset,
                               batch_size=config.batch_size,
                               shuffle=False,
                               num_workers=1,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)
    optimizer = optim_algorithm(model.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

    # Training criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    validation_criterion = nn.CrossEntropyLoss(ignore_index=0) # Validation criterion for classification


    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    config.cells = config.layers


    print('\nTraining started...\n')
    best_dev_acc = 0 # For discrete: higher is better
    best_dev_loss = None # For continuous: lower is better
    best_dev_epoch = 0

    for epoch in range(config.epochs):
        print(f"Epoch: {epoch+1}")
        best_dev_acc, best_dev_epoch = train(model, train_iter, dev_iter, optimizer, criterion, validation_criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch)

    test(model, test_iter, criterion, index_to_tag, device, config)

# --------------- FUNCTIONS FOR DISCRETE MODELS --------------------

def train(model, train_iterator, dev_iterator, optimizer, train_criterion, validation_criterion, index_to_tag, device, config, best_dev_acc, best_dev_epoch, epoch):
    model.train()
    for i, batch in enumerate(train_iterator):
        words, x, is_main_piece, tags, y, seqlens, _, _ = batch

        if config.model == 'WordMajority':
            model.collect_stats(x, y)
            continue

        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        logits, y_true_for_loss, _ = model(x, y) # logits: (N, T, VOCAB)

        if config.model == 'ClassEncodings':
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y_true_for_loss = y_true_for_loss.view(-1, y_true_for_loss.shape[-1])  # also (N*T, VOCAB)
            loss = train_criterion(logits.to(device), y_true_for_loss.to(device))
        else:
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y_true_for_loss = y_true_for_loss.view(-1)  # (N*T,)
            loss = train_criterion(logits.to(device), y_true_for_loss.to(device))

        loss.backward()
        optimizer.step()

        # Log training loss and perform validation
        if (i + 1) % config.log_every == 0 or (i + 1) == len(train_iterator):
            print(f"Epoch {epoch+1} Training step: {i+1}/{len(train_iterator)}, training loss: {loss.item():<.4f}")

            # Perform validation
            model.eval() # Set model to evaluation mode
            val_acc, val_loss = valid(model, dev_iterator, validation_criterion, index_to_tag, device, config)
            model.train() # Set model back to training mode

            print(f"Epoch {epoch+1} Validation results (step {i+1}): Acc: {val_acc:<5.2f}%, Loss: {val_loss:<.4f}")

            # Save best model based on validation accuracy
            if val_acc > best_dev_acc:
                best_dev_acc = val_acc
                best_dev_epoch = epoch + 1 # Update with current epoch
                dev_snapshot_path = 'best_model_{}_devacc_{}_epoch_{}.pt'.format(config.model, round(best_dev_acc, 2), best_dev_epoch)
                torch.save(model.state_dict(), dev_snapshot_path)
                print(f"New best model saved to {dev_snapshot_path}")

        # Save model every N iterations
        if (i + 1) % config.save_every_iterations == 0 or (i + 1) == len(train_iterator):
            snapshot_path = 'model_{}_epoch_{}_step_{}.pt'.format(config.model, epoch + 1, i + 1)
            torch.save(model.state_dict(), snapshot_path)
            print(f"Regular model saved to {snapshot_path}")

    if config.model == 'WordMajority':
        model.save_stats()

    return best_dev_acc, best_dev_epoch # Return updated best_dev_acc and best_dev_epoch


def valid(model, iterator, criterion, index_to_tag, device, config):
    model.eval()
    dev_losses = []
    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, _, _ = batch
            x = x.to(device)
            y = y.to(device)

            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits.to(device), labels.to(device))

            dev_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
        preds = [index_to_tag[hat] for hat in y_hat]

        if config.model != 'LSTM' and config.model != 'BiLSTM':
            tagslice = tags.split()[1:-1]
            predsslice = preds[1:-1]
            # assert len(preds) == len(words.split()) == len(tags.split())
        else:
            tagslice = tags.split()
            predsslice = preds
        for t, p in zip(tagslice, predsslice):
            if config.ignore_punctuation:
                if t != 'NA':
                    true.append(t)
                    predictions.append(p)
            else:
                true.append(t)
                predictions.append(p)

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    avg_dev_loss = np.mean(dev_losses)

    return acc, avg_dev_loss # Return acc and loss


def test(model, iterator, criterion, index_to_tag, device, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    # Load the best model for testing
    best_model_path_pattern = 'best_model_*_devacc_*.pt'
    best_model_files = glob.glob(best_model_path_pattern)
    if best_model_files:
        latest_best_model_path = max(best_model_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_best_model_path))
        print(f"Loaded best model from {latest_best_model_path} for testing.")
    else:
        print("No best model found to load for testing. Using current model state.")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, _, _ = batch
            x = x.to(device)
            y = y.to(device)

            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits, labels)

            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
            preds = [index_to_tag[hat] for hat in y_hat]
            if config.model != 'LSTM' and config.model != 'BiLSTM':
                tagslice = tags.split()[1:-1]
                predsslice = preds[1:-1]
                wordslice = words.split()[1:-1]
                # assert len(preds) == len(words.split()) == len(tags.split())
            else:
                tagslice = tags.split()
                predsslice = preds
                wordslice = words.split()
            for w, t, p in zip(wordslice, tagslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, t, p))
                if config.ignore_punctuation:
                    if t != 'NA':
                        true.append(t)
                        predictions.append(p)
                else:
                    true.append(t)
                    predictions.append(p)
            results.write("\n")

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)

    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    print('Test accuracy: {:<5.2f}%, Test loss: {:<.4f} after {} epochs.\n'.format(round(acc, 2), np.mean(test_losses),
                                                                                   config.epochs))

    final_snapshot_path = 'final_model_{}_testacc_{}_epoch_{}.pt'.format(config.model,
                                                                 round(acc, 2), config.epochs)
    torch.save(model.state_dict(), final_snapshot_path) # Save final model state_dict



# ---------------- FUNCTIONS FOR CONTINUOUS MODELS ------------------
''' These are used only the BertRegression and LSTMRegression models for now '''

def train_cont(model, train_iterator, dev_iterator, optimizer, train_criterion, validation_criterion, device, config, best_dev_loss, best_dev_epoch, epoch):

    model.train()
    for i, batch in enumerate(train_iterator):
        epoch=i
        words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch

        optimizer.zero_grad()
        x = x.to(device)
        values = values.to(device)

        predictions, true = model(x, values)
        loss = train_criterion(predictions.to(device), true.float().to(device))
        loss.backward()
        optimizer.step()

        # Log training loss and perform validation
        if (i + 1) % config.log_every == 0 or (i + 1) == len(train_iterator):
            print(f"Epoch {epoch+1} Training step: {i+1}/{len(train_iterator)}, training loss: {loss.item():<.4f}")

            # Perform validation
            model.eval() # Set model to evaluation mode
            _, val_loss = valid_cont(model, dev_iterator, validation_criterion, device, config)
            model.train() # Set model back to training mode

            print(f"Epoch {epoch+1} Validation results (step {i+1}): Loss: {val_loss:<.4f}")

            # For regression, 'best' is usually lowest loss.
            if best_dev_loss is None or val_loss < best_dev_loss: # Use best_dev_loss to track lowest validation loss
                best_dev_loss = val_loss # Update with current lowest loss
                best_dev_epoch = epoch + 1 # Update with current epoch
                dev_snapshot_path = 'best_model_{}_lowestloss_{}_epoch_{}.pt'.format(config.model, round(best_dev_loss, 4), best_dev_epoch)
                torch.save(model.state_dict(), dev_snapshot_path)
                print(f"New best model (lowest loss) saved to {dev_snapshot_path}")

        # Save model every N iterations
        if (i + 1) % config.save_every_iterations == 0 or (i + 1) == len(train_iterator):
            snapshot_path = 'model_{}_epoch_{}_step_{}.pt'.format(config.model, epoch + 1, i + 1)
            torch.save(model.state_dict(), snapshot_path)
            print(f"Regular model saved to {snapshot_path}")

    return best_dev_loss, best_dev_epoch # Return updated best_dev_loss and best_dev_epoch


def valid_cont(model, iterator, criterion, device, config):
    model.eval()
    dev_losses = []
    # No need for Words, Is_main_piece, Tags, Y, Predictions, Values for returning loss
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch
            x = x.to(device)
            values = values.to(device)

            predictions, true = model(x, values)
            loss = criterion(predictions.to(device), true.float().to(device))
            dev_losses.append(loss.item())

    avg_dev_loss = np.mean(dev_losses)
    return None, avg_dev_loss # No accuracy for continuous models


def test_cont(model, iterator, criterion, index_to_tag, device, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    # Load the best model for testing
    best_model_path_pattern = 'best_model_*_lowestloss_*.pt'
    best_model_files = glob.glob(best_model_path_pattern)
    if best_model_files:
        latest_best_model_path = max(best_model_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_best_model_path))
        print(f"Loaded best model from {latest_best_model_path} for testing.")
    else:
        print("No best model found to load for testing. Using current model state.")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Predictions, Values = [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens, values, invalid_set_to = batch
            x = x.to(device)
            values = values.to(device)

            predictions, true = model(x, values)
            loss = criterion(predictions.to(device), true.float().to(device))
            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Predictions.extend(predictions.cpu().numpy().tolist())
            Values.extend(values.cpu().numpy().tolist())

    true = []
    preds_to_eval = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, preds, values in zip(Words, Is_main_piece, Tags, Predictions, Values):
            valid_preds = [p for head, p in zip(is_main_piece, preds) if head == 1]

            predsslice = valid_preds[1:-1]
            valuesslice = values[1:-1]
            wordslice = words.split()[1:-1]

            for w, v, p in zip(wordslice, valuesslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, v, p))
                if v != invalid_set_to:
                    true.append(v)
                    preds_to_eval.append(p)
            results.write("\n")
    # calc metric
    y_true = np.array(true)
    y_pred = np.array(preds_to_eval)

    print('Test loss: {:<.4f}\n'.format(np.mean(test_losses)))
    # Correlation is calculated aftezrwards with a separate script.


if __name__ == "__main__":
    main()