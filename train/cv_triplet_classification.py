
import os
import sys

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

current_backbone_model = None
current_dataset = None


def save_model(model_path, model):
  """
  Save the model to the given path.

  Args:
    model_path: The path to save the model to.
    model: The model to save.
  """
  import torch

  # Ensure the model path is a string
  if not isinstance(model_path, str):
    raise ValueError("model_path must be a string")

  # Ensure the directory exists
  if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
  torch.save(model.state_dict(), model_path)
  print(f'Model saved to {model_path}')


def validate_model(model, dataset, batch_size=64):
  """
  Validate the model on the given dataset and return the average loss and accuracy.

  Args:
    model: The model to validate.
    dataset: The dataset to validate on.
    batch_size: The batch size to use for validation.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
  import torch

  device = get_device()
  model = model.to(device)
  model.eval()

  loss_fn = torch.nn.CrossEntropyLoss()
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  total_loss = 0.0
  correct_predictions = 0
  total_samples = 0

  y_true = []
  y_pred = []

  with torch.no_grad():
    for inputs, labels in data_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      total_loss += loss.item()

      # Calculate accuracy
      _, predicted = torch.max(outputs.data, 1)
      total_samples += labels.size(0)
      correct_predictions += (predicted == labels).sum().item()

      # Store true and predicted labels for metrics
      y_true += labels.tolist()
      y_pred += predicted.tolist()

  # Calculate average loss and accuracy
  avg_loss = total_loss / len(data_loader)
  # accuracy = (correct_predictions / total_samples) * 100
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
  accuracy = accuracy_score(y_true, y_pred)
  # Print the validation results
  print(
      f'Validation average loss: {avg_loss}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%')
  return avg_loss, accuracy, precision, recall, f1


def find_k_nearest_semantic_embeddings(knowledge_dataset, target_embeddings, k=3, batch_size=64):
  """
  Find the k nearest semantic embeddings for each embedding in the dataset.
  Args:
    knowledge_dataset: A dataset containing semantic embeddings to search in.
    target_embeddings: The embeddings to find the nearest neighbors for.
    k: The number of nearest neighbors to find.
  Returns:
    A list of indices of the k nearest semantic embeddings.
  """
  from trainer import get_device
  import torch
  from torch.utils.data import DataLoader

  device = get_device()

  # Collect knowledge dataset embeddings and labels
  k_embeddings_list = []
  k_labels_list = []
  knowledge_dataloader = DataLoader(knowledge_dataset, batch_size=batch_size, shuffle=False)

  # Extract embeddings from knowledge dataset
  for k_embeddings, k_labels in knowledge_dataloader:
    k_embeddings_list.append(k_embeddings.to(device))
    k_labels_list.extend([str(label.item()) for label in k_labels])

  # Stack all knowledge embeddings into a single tensor for efficient computation
  if not k_embeddings_list:
    raise ValueError("Knowledge dataset is empty")
  k_embeddings_tensor = torch.cat(k_embeddings_list).to(device)

  # Compute pairwise distances efficiently (batch computation)
  distances = torch.cdist(torch.stack(target_embeddings), k_embeddings_tensor)
  # Get indices of k nearest neighbors for each target embedding
  _, indices = torch.topk(distances, k, dim=1, largest=False)

  return indices


def semantic_validate_model(model, validate_dataset, knowledge_dataset, label_to_embeddings, batch_size=64):
  """
    Validate the model on the given dataset using semantic embeddings and return the average loss and accuracy.
  Args:
    model: The model to validate (model must be Classifier).
    validate_dataset: The dataset to validate on (already contains embeddings).
    knowledge_dataset: The dataset containing knowledge embeddings (already contains embeddings).
    label_to_embeddings: Dictionary mapping labels to their semantic embeddings.
    batch_size: The batch size to use for validation.
  """

  from trainer import get_device
  from torch.utils.data import DataLoader
  from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
  import torch
  import pandas as pd

  device = get_device()
  model = model.to(device)
  model.eval()
  loss_fn = torch.nn.CrossEntropyLoss()
  validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
  knowledge_dataloader = DataLoader(knowledge_dataset, batch_size=batch_size, shuffle=False)

  total_loss = 0.0
  correct_predictions = 0
  total_samples = 0

  num_of_k_nearest = 3

  y_true = []
  y_pred = []

  with torch.no_grad():
    # Collect knowledge dataset embeddings and labels
    k_embeddings_list = []
    k_labels_list = []

    # Extract embeddings from knowledge dataset
    for k_embeddings, k_labels in knowledge_dataloader:
      k_embeddings_list.append(k_embeddings.to(device))
      k_labels_list.extend([str(label.item()) for label in k_labels])

    # Stack all knowledge embeddings into a single tensor for efficient computation
    if not k_embeddings_list:
      raise ValueError("Knowledge dataset is empty")

    k_embeddings_tensor = torch.cat(k_embeddings_list).to(device)

    # Now validate on the validation dataset
    for val_embeddings, labels in validate_dataloader:
      val_embeddings, labels = val_embeddings.to(device), labels.to(device)
      # Compute pairwise distances efficiently (batch computation)
      # Shape: [val_embeddings.size(0), k_embeddings_tensor.size(0)]
      distances = torch.cdist(val_embeddings, k_embeddings_tensor)

      # Get indices of k nearest neighbors for each validation embedding
      _, indices = torch.topk(distances, num_of_k_nearest, dim=1, largest=False)

      # Prepare label embeddings tensor once
      label_embeddings_tensor = torch.stack([label_to_embeddings[label].to(device) for label in k_labels_list])

      # Gather the embeddings of the nearest neighbors
      # For each validation point, get its k nearest neighbors' embeddings
      batch_size, k = indices.size()
      nearest_embeddings = label_embeddings_tensor[indices.view(-1)].view(batch_size, k, -1)

      # Calculate centroids for each validation point
      centroid_embeddings = torch.mean(nearest_embeddings, dim=1)

      # move the centroid embeddings to the same device as the model
      val_embeddings -= centroid_embeddings

      # # Calculate the label embeddings for the current batch
      # label_embeddings = torch.stack([label_to_embeddings[str(label.item())].detach().to(device) for label in labels])
      # # Subtract the label embeddings from the inputs
      # val_embeddings = val_embeddings - label_embeddings

      # Use the transformed embeddings for final classification
      outputs = model(val_embeddings)  # For Classifier models

      loss = loss_fn(outputs, labels)
      total_loss += loss.item()
      # Calculate accuracy
      _, predicted = torch.max(outputs.data, 1)
      total_samples += labels.size(0)
      correct_predictions += (predicted == labels).sum().item()
      # Store true and predicted labels for metrics
      y_true += labels.tolist()
      y_pred += predicted.tolist()

  # Calculate precision, recall, and F1 score
  precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
  recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
  f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
  accuracy = accuracy_score(y_true, y_pred)
  # Calculate average loss and accuracy
  avg_loss = total_loss / len(validate_dataloader)
  # accuracy = (correct_predictions / total_samples) * 100
  print(
      f'Semantic Validation average loss: {avg_loss}, '
      f'Accuracy: {100 * accuracy:.2f}%, '
      f'Precision: {100 * precision:.2f}%, '
      f'Recall: {100 * recall:.2f}%, '
      f'F1 Score: {100 * f1:.2f}%')
  return avg_loss, accuracy, precision, recall, f1


def compute_label_embeddings(labels, out_features):
  """ Compute label encodings for the given labels using the specified model.
  Args:
    labels: A list of labels to encode.
    model: The model to use for encoding the labels.
  Returns:
    A list of encoded labels.
  """

  import torch
  import torch.nn.functional as F
  from transformers import AutoTokenizer, AutoModel

  labels = sorted(set(labels))  # Ensure unique labels
  # convert labels to strings if they are not already
  labels = [str(label) for label in labels]
  # Load the tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  nomic = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
  # encode labels to embeddings
  labels_embeddings = tokenizer(labels, padding=True, truncation=True, return_tensors='pt')

  with torch.no_grad():
    embeddings = nomic(**labels_embeddings)

  # max pooling the label embeddings
  token_embeddings = embeddings[0]
  attention_mask = labels_embeddings['attention_mask']

  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
  # scale the embeddings to out_features
  embeddings = torch.nn.Linear(embeddings.shape[1], out_features)(embeddings)
  # normalize the embeddings
  embeddings = F.normalize(embeddings, p=2, dim=1)
  # Store the embeddings, the labels_embeddings should have absolute values
  labels_embeddings = abs(embeddings)

  # create a dictionary to map labels to embeddings
  label_to_embedding = {label: embedding for label, embedding in zip(labels, labels_embeddings)}
  return label_to_embedding


def get_n_most_informative_samples_indices(dataset, model, samples_taken_indices: list, n=2):
  """
  Get the n most informative samples from the dataset using the model.
  Args:
    dataset: The dataset to get samples from.
    model: The model to use for getting the samples.
    n: The number of samples to return.
  """
  from torch.utils.data import DataLoader
  import torch
  import numpy as np
  from trainer import get_device

  device = get_device()
  # Create a data loader for the dataset
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  outputs = []
  # Get the embeddings for all samples in the dataset
  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      inputs = batch[0].to(device)
      outputs.append(model(inputs).cpu())
  outputs = torch.cat(outputs)
  # Calculate entropy to find the most informative samples
  with torch.no_grad():
    # Apply softmax to get probability distributions
    probs = torch.nn.functional.softmax(outputs, dim=1)

    # Calculate entropy for each sample: -sum(p * log(p))
    # Handle zero probabilities by adding a small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)

    # Sort samples by entropy (higher entropy = more uncertainty = more informative)
    # We use numpy for sorting to get indices that we can use to select original samples
    entropy_np = entropy.cpu().numpy()
    # exclude already taken samples
    entropy_np[samples_taken_indices] = -np.inf  # Set taken samples' entropy to -inf to exclude them
    most_uncertain_indices = np.argsort(entropy_np)[::-1][:n]

  # Return the indices of the n most informative samples
  # These indices correspond to the original dataset
  return most_uncertain_indices.tolist()


def triplet_train_process(dataset, model, k_fold=5, batch_size=64):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch

  device = get_device()

  # split dataset using k-fold cross-validation
  dataset_size = len(dataset)
  fold_size = dataset_size // k_fold

  for fold in range(k_fold):
    print(f'Running fold {fold + 1}/{k_fold}...')

    # Initialize the trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.TripletMarginLoss(margin=0.2, p=2)

    # Split dataset into train and validation sets
    train_dataset = torch.utils.data.Subset(dataset, list(
        range(fold_size * fold)) + list(range(fold_size * (fold + 1), dataset_size)))
    val_dataset = torch.utils.data.Subset(dataset, range(fold_size * fold, fold_size * (fold + 1)))
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    epochs = 10
    loss_list = []
    test_loss_list = []
    # loop through epochs
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}...')
      start_time = time.time()
      model = model.to(device)
      model.train()
      total_loss = 0.0

      # loop through batches and train
      for batch in train_loader:
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(
            device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        output_anchor = model(anchor)
        output_positive = model(positive)
        output_negative = model(negative)

        loss = loss_fn(output_anchor, output_positive, output_negative)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      # loop through test set to evaluate
      model.eval()
      total_test_loss = 0.0
      with torch.no_grad():
        for batch in val_loader:
          anchor, positive, negative = batch
          anchor, positive, negative = anchor.to(
              device), positive.to(device), negative.to(device)

          output_anchor = model(anchor)
          output_positive = model(positive)
          output_negative = model(negative)

          loss = loss_fn(output_anchor, output_positive, output_negative)
          total_test_loss += loss.item()

      # Calculate average loss for the epoch
      avg_loss = total_loss / len(train_loader)
      avg_test_loss = total_test_loss / len(val_loader)
      print(
          f'Fold {fold + 1}: Average loss: {avg_loss}, Average test loss: {avg_test_loss}')
      print(
          f'Time taken for fold {fold + 1}, epoch {epoch + 1}: {time.time() - start_time:.2f} seconds')
      loss_list.append(avg_loss)
      test_loss_list.append(avg_test_loss)
    print(f'Fold {fold + 1} completed.')

    # save the model after each fold
    model_dir = f'models/triplet/{current_dataset}_{model._get_name()}'
    save_model(f'{model_dir}/model_fold_{fold + 1}.pth', model)

  # Print the average loss over all folds
  average_loss = sum(loss_list) / len(loss_list)
  average_test_loss = sum(test_loss_list) / len(test_loss_list)
  print(
      f'Over all folds: Average loss : {average_loss}, Average test loss: {average_test_loss}')

  # Return the trained model and the average loss
  return model, average_loss, average_test_loss


def AL_RL_semantic_classification_train_process(dataset, model, semantic_embedding_fn, k_fold=5, batch_size=64, test_dataset=None):
  """
  Train the model on the given dataset and return the scored model and average loss.
  This function is a placeholder for a specific training process that might involve moving labels to function.
  Args:
    dataset: The dataset to train on (ds.EmbeddedDataset).
    model: The model to train.
    semantic_embedding_fn: A function to compute semantic embeddings for the labels.
    k_fold: The number of folds for cross-validation.
    batch_size: The batch size to use for training.
    test_dataset: Optional; a dataset to evaluate the model on after training.
  Returns:
    The trained model and the average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch
  import pandas as pd
  from model import DDQNAgent

  class AlClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
      self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
      return len(self.dataframe)

    def __getitem__(self, idx):
      row = self.dataframe.iloc[idx]
      input_tensor = row['input']
      semantic_tensor = row['semantic']
      label_tensor = row['label']
      return input_tensor, semantic_tensor, label_tensor

  device = get_device()
  # create result df
  result_df = pd.DataFrame(columns=['dataset', 'model', 'fold', 'avg_loss', 'avg_val_loss',
                                    'avg_test_loss', 'precision', 'recall', 'f1', 'accuracy'])
  # load results from file if exists
  if os.path.exists(f'results/cv-triplet/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'):
    result_df = pd.read_csv(
        f'results/cv-triplet/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv')
  # compute label embeddings
  print('Computing label embeddings...')
  label_to_embedding = compute_label_embeddings(dataset.labels, model.backbone_out_features)
  if not semantic_embedding_fn:
    # Default to zero if no function is provided, it means no semantic embeddings are used
    def semantic_embedding_fn(x): return 0
  label_to_embedding = {label: semantic_embedding_fn(
      index) * embedding for index, (label, embedding) in enumerate(label_to_embedding.items())}
  print('Label embeddings computed.')

  # Initialize context for training
  epsilon = 0.001
  count_list = []  # To keep track of the number of actions taken in each fold
  num_episodes = 5
  num_al_iterations = 5
  num_epochs = 10
  num_of_k_nearest = 3

  # split dataset using k-fold cross-validation
  dataset_size = len(dataset)
  fold_size = dataset_size // k_fold
  for fold in range(k_fold):
    print(f'Running fold {fold + 1}/{k_fold}...')
    fold_start_time = time.time()
    # Do the same process as classification_train_process but with semantic embeddings
    semantic_embeddings = label_to_embedding
    # Split dataset into train and validation sets
    train_dataset = torch.utils.data.Subset(dataset, list(
        range(fold_size * fold)) + list(range(fold_size * (fold + 1), dataset_size)))
    val_dataset = torch.utils.data.Subset(dataset, range(fold_size * fold, fold_size * (fold + 1)))
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the agent
    agent = DDQNAgent(num_actions=len(dataset.labels), model=model, batch_size=batch_size, device=device)

    # Train the model
    for episode in range(num_episodes):  # Number of episodes
      print(f'Episode {episode + 1}/{num_episodes}...')
      count = 0
      train_df = pd.DataFrame(columns=['input', 'semantic', 'label'])
      # Train the agent
      for (inputs, labels) in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # get the semantic embeddings for the labels
        labels_embeddings = torch.stack([semantic_embeddings[str(label.item())] for label in labels]).to(device)
        # append the inputs, semantic embeddings and labels to the train_df
        train_data_list = []
        for i in range(len(inputs)):
          train_data_list.append({
              'input': inputs[i].cpu().detach().numpy(),
              'semantic': labels_embeddings[i].cpu().detach().numpy(),
              'label': labels[i].cpu().detach().numpy()
          })
        train_df = pd.concat([train_df, pd.DataFrame(train_data_list)], ignore_index=True)

        # For each input, select an action using the agent
        for input_index in range(len(inputs)):
          state = inputs[input_index]  # Add batch dimension
          # select an action using the agent
          action = agent.select_action(state.unsqueeze(0), epsilon=epsilon)  # Epsilon-greedy action selection
          if input_index < len(inputs) - 2:
            next_state = inputs[input_index + 1]  # Next state
            if action != labels[input_index].item():
              reward = -1  # Negative reward for incorrect action
              done = False
            else:
              reward = 1  # Positive reward for correct action
              done = True
            agent.store_experience(state, action, reward, next_state, done)
      count_list.append(count)

      # Train the agent using the stored experiences
      _, _, train_accuracy = agent.train_q_net(train_loader, semantic_embeddings)
      print(f'Training accuracy for episode {episode + 1}: {train_accuracy:.2f}%')
      print('Before Update: ', end='')
      semantic_validate_model(agent.q_net, val_dataset, train_dataset, label_to_embedding, batch_size=batch_size)
      # Update agent's q-network if enough experiences are stored
      if len(agent.memory) > agent.batch_size:
        agent.update()
      print('After Update: ', end='')
      semantic_validate_model(agent.q_net, val_dataset, train_dataset, label_to_embedding, batch_size=batch_size)

      ######################### Active Learning Iterations #########################
      sample_taken = 0
      samples_taken_indices = []  # To keep track of the indices of samples taken in each iteration
      # apply active learning logic here
      for al_iteration in range(num_al_iterations):  # Number of active learning iterations
        print(f'Active Learning iteration {al_iteration + 1}/{num_al_iterations}...')
        sample_taken += 2
        # Get the n most informative samples indices
        informative_samples_indices = get_n_most_informative_samples_indices(
            val_dataset, agent.q_net, samples_taken_indices, n=sample_taken)
        # Update the samples taken indices
        samples_taken_indices.extend(informative_samples_indices)
        print(f'Informative samples indices: {informative_samples_indices}')
        # Get the informative samples from the validation dataset
        agent.q_net.eval()  # Set the model to evaluation mode
        with torch.no_grad():
          informative_samples = [val_dataset[i] for i in informative_samples_indices]
          # append the informative samples to the train_df, but use predicted labels from agent
          train_data_list = []
          sample_embeddings = [sample[0] for sample in informative_samples]
          # Find k nearest semantic embeddings
          k_nearest_indices = find_k_nearest_semantic_embeddings(train_dataset, sample_embeddings, k=num_of_k_nearest)
          # get the semantic embeddings for the k nearest neighbors each sample
          k_nearest_semantic_embeddings = [
              [semantic_embeddings[str(train_dataset[j][1])] for j in indices] for indices in k_nearest_indices
          ]
          # calculate the mean of the k nearest semantic embeddings
          k_nearest_semantic_embeddings = [
              torch.mean(torch.stack(embeddings), dim=0) for embeddings in k_nearest_semantic_embeddings
          ]
          # Create a list of tuples (input, predicted semantic embedding, label)
          # where input is the true input, predicted semantic embedding is the mean of the k nearest semantic embeddings,
          # and label is the true label from the informative samples
          informative_samples = zip(
              [sample[0] for sample in informative_samples],  # true inputs
              k_nearest_semantic_embeddings,  # predicted semantic embeddings
              [sample[1] for sample in informative_samples]  # true labels
          )
          for (is_input, is_semen, is_label) in informative_samples:
            train_data_list.append({
                'input': is_input.cpu().detach().numpy(),
                'semantic': is_semen.cpu().detach().numpy(),
                'label': is_label
            })
          # Append the new samples to the train_df
          train_df = pd.concat([train_df, pd.DataFrame(train_data_list)], ignore_index=True)

        # Initialize the AL Train Configuration
        al_criterion = torch.nn.CrossEntropyLoss()
        al_optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=1e-4)
        al_train_loss_list = []

        # Convert the train_df to a DataLoader
        al_train_loader = DataLoader(AlClassificationDataset(train_df), batch_size=batch_size, shuffle=False)
        prev_train_accuracy = 0.0
        for epoch in range(num_epochs):
          model.train()
          al_total = 0
          al_correct = 0
          al_total_loss = 0.0
          for (inputs, semans, labels) in al_train_loader:
            inputs, semans, labels = inputs.to(device), semans.to(device), labels.to(device)
            al_optimizer.zero_grad()
            inputs = inputs - semans
            outputs = agent.q_net(inputs)
            _, al_predicted = torch.max(outputs.data, 1)
            al_correct += (al_predicted == labels).sum().item()
            al_total += labels.size(0)
            al_loss = al_criterion(outputs, labels)
            al_loss.backward()
            al_optimizer.step()
            al_total_loss += al_loss.item()

          al_train_accuracy = 100 * al_correct / al_total
          al_train_loss_list.append(al_total_loss / len(al_train_loader))
          if prev_train_accuracy != 0.0 and al_train_accuracy != prev_train_accuracy:
            print()
          print(f'\rAL Iteration {al_iteration + 1}, Epoch {epoch + 1}/{num_epochs}... Loss: {al_total_loss / len(al_train_loader):.4f}, Accuracy: {al_train_accuracy:.2f}%', end='', flush=True)
          prev_train_accuracy = al_train_accuracy
        # New line after the last print statement
        print()

        # Validate the model after active learning iteration
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = semantic_validate_model(
            agent.q_net, val_dataset, train_dataset, label_to_embedding, batch_size=batch_size)
        print(f'AL Iteration {al_iteration + 1} Validation results:')
        print(f' - Loss: {avg_val_loss:.4f}')
        print(f' - Accuracy: {100*val_accuracy:.2f}%')
        print(f' - Precision: {100*val_precision:.2f}%')
        print(f' - Recall: {100*val_recall:.2f}%')
        print(f' - F1 Score: {100*val_f1:.2f}%')

    fold_end_time = time.time()

    # Validate the model
    avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = semantic_validate_model(
        model, val_dataset, train_dataset, label_to_embedding, batch_size=batch_size)
    print(f'Validation results for fold {fold + 1}:')
    print(f' - Loss: {avg_val_loss:.4f}')
    print(f' - Accuracy: {100*val_accuracy:.2f}%')
    print(f' - Precision: {100*val_precision:.2f}%')
    print(f' - Recall: {100*val_recall:.2f}%')
    print(f' - F1 Score: {100*val_f1:.2f}%')

    # Save the model after each fold
    model_dir = f'models/classification/{current_dataset}_{current_backbone_model._get_name()}'
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    model_path = f'{model_dir}/model_fold_{fold + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # validate the model on the test set each fold if provided
    if test_dataset is not None:
      avg_test_loss, accuracy, precision, recall, f1 = semantic_validate_model(
          model, test_dataset, train_dataset, label_to_embedding, batch_size=batch_size)
      print(f'Fold {fold + 1}: Average test loss: {avg_test_loss}, Test accuracy: {100 * accuracy:.2f}%, Test precision: {100 * precision:.2f}%, Test recall: {100 * recall:.2f}%, Test F1 Score: {100 * f1:.2f}%')
      result_df = pd.concat([result_df, pd.DataFrame({
          'dataset': current_dataset,
          'model': model._get_name(),
          'fold': fold + 1,
          'avg_loss': 0,
          'avg_test_loss': avg_test_loss,
          'avg_val_loss': avg_val_loss,
          'precision': precision,
          'recall': recall,
          'f1': f1,
          'accuracy': accuracy,
          'total_time': fold_end_time - fold_start_time
      }, index=[0])], ignore_index=True)

  # save the results to a CSV file to keep track of the training process by current_backbone_model and current_dataset
  if current_backbone_model is not None and current_dataset is not None:
    result_df['dataset'] = current_dataset
  if not result_df.empty:
    result_dir = 'results/cv-triplet'
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)
    result_path = f'{result_dir}/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # If the directory does not exist, create it
    # save the results to a CSV file with the current dataset and model in name and date
    result_df.to_csv(result_path, index=False)

  # Return the trained model and the average loss
  return model, 0, 0, label_to_embedding


def semantic_classification_train_process(dataset, model, semantic_embedding_fn, k_fold=5, batch_size=64, test_dataset=None):
  """ Train the model on the given dataset and return the scored model and average loss.
  This function is a placeholder for a specific training process that might involve moving labels to function.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch
  import pandas as pd

  device = get_device()

  # create result df
  result_df = pd.DataFrame(columns=['dataset', 'model', 'fold', 'avg_loss', 'avg_test_loss',
                           'avg_val_loss', 'precision', 'recall', 'f1', 'accuracy'])
  # load results from file if exists
  # f'{result_dir}/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
  if os.path.exists(f'results/cv-triplet/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'):
    result_df = pd.read_csv(
        f'results/cv-triplet/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv')

  # compute label embeddings
  print('Computing label embeddings...')
  label_to_embedding = compute_label_embeddings(dataset.labels, model.backbone_out_features)
  if not semantic_embedding_fn:
    # Default to zero if no function is provided, it means no semantic embeddings are used
    def semantic_embedding_fn(x): return 0
  label_to_embedding = {label: semantic_embedding_fn(
      index) * embedding for index, (label, embedding) in enumerate(label_to_embedding.items())}
  print('Label embeddings computed.')

  # split dataset using k-fold cross-validation
  dataset_size = len(dataset)
  fold_size = dataset_size // k_fold

  for fold in range(k_fold):
    # # Check if fold already exists in results
    # is_fold_trained = True
    # if fold in result_df['fold'].values:
    #   print(f'Fold {fold + 1} already exists in results.')
    #   # try loading the model
    #   model_path = f'models/classification/{current_dataset}_{model._get_name()}/model_fold_{fold + 1}.pth'
    #   if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    #     print(f'Model loaded from {model_path}')
    #   else:
    #     # If the model does not exist, retrain the fold
    #     is_fold_trained = False

    # if is_fold_trained:
    #   pass
    print(f'Running fold {fold + 1}/{k_fold}...')
    fold_start_time = time.time()

    # Do the same process as classification_train_process but with semantic embeddings
    semantic_embeddings = label_to_embedding
    # Initialize the trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Split dataset into train and validation sets
    train_dataset = torch.utils.data.Subset(dataset, list(
        range(fold_size * fold)) + list(range(fold_size * (fold + 1), dataset_size)))
    val_dataset = torch.utils.data.Subset(dataset, range(fold_size * fold, fold_size * (fold + 1)))
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    epochs = 10
    loss_list = []
    val_loss_list = []
    # loop through epochs
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}...')
      epoch_start_time = time.time()
      model = model.to(device)
      model.train()
      total_loss = 0.0
      # loop through batches and train
      for batch in train_loader:
        inputs, labels = batch

        # move input embeddings to semantic embeddings
        label_embeddings = torch.stack([semantic_embeddings[str(label.item())].detach() for label in labels])
        # subtract the label embeddings from the inputs
        inputs = inputs - label_embeddings.to(device)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      epoch_end_time = time.time()

      # loop through validation set to evaluate
      avg_loss = total_loss / len(train_loader)
      avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = semantic_validate_model(
          model, val_dataset, train_dataset, label_to_embedding, batch_size=batch_size)
      print(f'Fold {fold + 1}, Epoch {epoch + 1}: Average loss: {avg_loss}, Validation loss: {avg_val_loss}, Validation accuracy: {val_accuracy:.2f}%, Time taken: {epoch_end_time - epoch_start_time:.2f} seconds')
      loss_list.append(avg_loss)
      val_loss_list.append(avg_val_loss)

    fold_end_time = time.time()
    print(f'Fold {fold + 1} completed.')
    # save the model after each fold
    model_dir = f'models/classification/{current_dataset}_{model._get_name()}'
    save_model(f'{model_dir}/model_fold_{fold + 1}.pth', model)
    # Validate the model on the test set each fold if provided
    if test_dataset is not None:
      avg_test_loss, accuracy, precision, recall, f1 = semantic_validate_model(
          model, test_dataset, train_dataset, label_to_embedding, batch_size=batch_size)
      print(f'Fold {fold + 1}: Average test loss: {avg_test_loss}, Test accuracy: {accuracy:.2f}%, Test precision: {precision:.2f}%, Test recall: {recall:.2f}%, Test F1 Score: {f1:.2f}%')
      result_df = pd.concat([result_df, pd.DataFrame({
          'dataset': current_dataset,
          'model': model._get_name(),
          'fold': fold + 1,
          'avg_loss': total_loss / len(train_loader),
          'avg_test_loss': avg_test_loss,
          'avg_val_loss': avg_val_loss,
          'precision': precision,
          'recall': recall,
          'f1': f1,
          'accuracy': accuracy,
          'total_time': fold_end_time - fold_start_time
      }, index=[0])], ignore_index=True)

  # Print the average loss over all folds
  average_loss = sum(loss_list) / len(loss_list)
  average_val_loss = sum(val_loss_list) / len(val_loss_list)
  print(
      f'Over all folds: Average loss : {average_loss}, Average validation loss: {average_val_loss}')

  # save the results to a CSV file to keep track of the training process by current_backbone_model and current_dataset
  if current_backbone_model is not None and current_dataset is not None:
    result_df['dataset'] = current_dataset
  if not result_df.empty:
    result_dir = 'results/cv-triplet'
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)
    result_path = f'{result_dir}/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # If the directory does not exist, create it
    # save the results to a CSV file with the current dataset and model in name and date
    result_df.to_csv(result_path, index=False)

  # Return the trained model and the average loss
  return model, average_loss, average_val_loss, label_to_embedding


def classification_train_process(dataset, model, k_fold=5, batch_size=64, test_dataset=None):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch
  import pandas as pd

  device = get_device()

  # create result df
  result_df = pd.DataFrame(columns=['dataset', 'model', 'fold', 'avg_loss', 'avg_test_loss',
                           'avg_val_loss', 'precision', 'recall', 'f1', 'accuracy'])

  for fold in range(k_fold):
    print(f'Running fold {fold + 1}/{k_fold}...')
    fold_start_time = time.time()
    # Initialize the trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Train the model
    epochs = 10
    loss_list = []
    val_loss_list = []
    # loop through epochs
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}...')
      epoch_start_time = time.time()
      model = model.to(device)
      model.train()
      total_loss = 0.0

      # loop through batches and train
      for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      epoch_end_time = time.time()
      # loop through test set to evaluate
      model.eval()
      total_val_loss = 0.0
      with torch.no_grad():
        for batch in val_loader:
          inputs, labels = batch
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)
          loss = loss_fn(outputs, labels)
          total_val_loss += loss.item()

      # Calculate average loss for the epoch
      avg_loss = total_loss / len(train_loader)
      avg_val_loss = total_val_loss / len(val_loader)
      print(
          f'Fold {fold + 1}: Average loss: {avg_loss}, Average validation loss: {avg_val_loss}')
      print(
          f'Time taken for fold {fold + 1}, epoch {epoch + 1}: {epoch_end_time - epoch_start_time:.2f} seconds')
      loss_list.append(avg_loss)
      val_loss_list.append(avg_val_loss)
    print(f'Fold {fold + 1} completed.')

    # save the model after each fold
    model_dir = f'models/classification/{current_dataset}_{model._get_name()}'
    save_model(f'{model_dir}/model_fold_{fold + 1}.pth', model)

    # Validate the model on the test set each fold if provided
    if test_dataset is not None:
      avg_test_loss, accuracy, precision, recall, f1 = validate_model(
          model, test_dataset, batch_size=batch_size)
      print(
          f'Fold {fold + 1}: Average test loss: {avg_test_loss}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%')
      result_df = pd.concat([result_df, pd.DataFrame({
          'dataset': [dataset.__class__.__name__],
          'model': [model._get_name()],
          'fold': [fold + 1],
          'avg_loss': [avg_loss],
          'avg_test_loss': [avg_test_loss],
          'avg_val_loss': [avg_val_loss],
          'accuracy': [accuracy],
          'precision': [precision],
          'recall': [recall],
          'f1': [f1],
          'total_time': [time.time() - fold_start_time]
      })], ignore_index=True)

  # Print the average loss over all folds
  average_loss = sum(loss_list) / len(loss_list)
  average_val_loss = sum(val_loss_list) / len(val_loss_list)
  print(
      f'Over all folds: Average loss : {average_loss}, Average validation loss: {average_val_loss}')

  # save the results to a CSV file to keep track of the training process by current_backbone_model and current_dataset
  if current_backbone_model is not None and current_dataset is not None:
    result_df['dataset'] = current_dataset
  if not result_df.empty:
    result_dir = 'results/cv-triplet'
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)
    result_path = f'{result_dir}/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # If the directory does not exist, create it
    # save the results to a CSV file with the current dataset and model in name and date
    result_df.to_csv(result_path, index=False)

  # Return the trained model and the average loss
  return model, average_loss, average_val_loss


def train(dataset, model, train_process='triplet', semantic_embedding_fn=None, k_fold=5, batch_size=32, test_dataset=None):
  """
  Train the model on the given dataset and return the scored model and average loss.

  Args:
    dataset: The dataset to train on.
    model: The model to train.
    train_process: The training process to use, either 'triplet' or 'classification' or 'semantic_classification'.
    k_fold: The number of folds for k-fold cross-validation.
    batch_size: The batch size to use for training.
  """
  print('Starting training process...')
  if train_process == 'triplet':
    trained_model, avg_loss, avg_test_loss = triplet_train_process(
        dataset, model, k_fold=k_fold, batch_size=batch_size)
  elif train_process == 'classification':
    trained_model, avg_loss, avg_test_loss = classification_train_process(
        dataset, model, k_fold=k_fold, batch_size=batch_size, test_dataset=test_dataset)
  elif train_process == 'semantic_classification':
    trained_model, avg_loss, avg_test_loss, label_to_embedding = semantic_classification_train_process(
        dataset, model, semantic_embedding_fn=semantic_embedding_fn, k_fold=k_fold, batch_size=batch_size, test_dataset=test_dataset)
  elif train_process == 'AL_RL_semantic_classification':
    trained_model, avg_loss, avg_test_loss, label_to_embedding = AL_RL_semantic_classification_train_process(
        dataset, model, semantic_embedding_fn=semantic_embedding_fn, k_fold=k_fold, batch_size=batch_size, test_dataset=test_dataset)
  else:
    raise ValueError(f'Unknown training process: {train_process}')
  print('Training completed.')
  label_to_embedding = label_to_embedding if train_process in [
      'semantic_classification', 'AL_RL_semantic_classification'] else None
  return trained_model, avg_loss, avg_test_loss, label_to_embedding


def create_training_process_df(
        dataset_type,
        create_triplet_dataset_fn,
        create_classification_dataset_fn,
        create_classification_test_dataset_fn=None,
        models: list[str] = ['resnet'],
        batch_size=32):
  """
  Create a DataFrame to store the training process information.
  Args:
    dataset_type: The type of dataset to use (e.g., 'cifar10', 'mnist').
    create_triplet_dataset_fn: Function to create the triplet dataset.
    create_classification_dataset_fn: Function to create the classification dataset.
    create_classification_test_dataset_fn: Function to create the classification test dataset (optional).
    models: List of model names to include in the DataFrame (e.g., ['resnet', 'vgg', 'mobilenet', 'densenet']).
    batch_size: The batch size to use for training.
  Returns:
    A DataFrame containing the training process information.
  """
  import pandas as pd
  import torchvision

  triplet_df = pd.DataFrame(columns=[
      'backbone_model',
      'feature_extractor_model',
      'dataset_type',
      'create_triplet_dataset_fn',
      'create_classification_dataset_fn',
      'create_classification_test_dataset_fn',
      'batch_size'
  ])

  # from modes, create a list of functions to create backbone models
  create_backbone_model_funcs = []
  for model in models:
    if model == 'resnet':
      create_backbone_model_funcs.append(lambda: torchvision.models.resnet50(weights=None))
    elif model == 'vgg':
      create_backbone_model_funcs.append(lambda: torchvision.models.vgg16(weights=None))
    elif model == 'mobilenet':
      create_backbone_model_funcs.append(lambda: torchvision.models.mobilenet_v2(weights=None))
    elif model == 'densenet':
      create_backbone_model_funcs.append(lambda: torchvision.models.densenet121(weights=None))
    else:
      raise ValueError(f'Unknown model type: {model}')

  # Add all backbone models to the DataFrame
  for create_fn in create_backbone_model_funcs:
    triplet_df = pd.concat([triplet_df, pd.DataFrame({
        'backbone_model': [create_fn()],
        'feature_extractor_model': [FeatureExtractor(create_fn())],
        'dataset_type': [dataset_type],
        'create_triplet_dataset_fn': [create_triplet_dataset_fn],
        'create_classification_dataset_fn': [create_classification_dataset_fn],
        'create_classification_test_dataset_fn': [create_classification_test_dataset_fn],
        'batch_size': [batch_size]
    })], ignore_index=True)

  return triplet_df


def create_train_test_dataset(create_train_dataset_fn, create_test_dataset_fn=None):
  """
  Create a train and test dataset from the given functions.
  """
  import torch

  # Create the train dataset
  train_dataset = create_train_dataset_fn()

  # If a test dataset function is provided, use it to create the test dataset
  if create_test_dataset_fn is not None:
    test_dataset = create_test_dataset_fn()
  else:
    # Otherwise, split the train dataset into train and test sets
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size])

  return train_dataset, test_dataset

# Helper function to train and evaluate a model


def train_and_evaluate(model, train_data, test_data, process_type, semantic_fn=None, description=""):
  """
  Train and evaluate a model, then record the results.
  This function checks if the model has already been trained on the dataset and loads it if available.
  If not, it trains the model and saves the results to a CSV file.
  Args:
    model: The model to train and evaluate.
    train_data: The training dataset.
    test_data: The test dataset.
    process_type: The type of training process ('triplet', 'classification', 'semantic_classification', 'AL_RL_semantic_classification').
    semantic_fn: A function to compute semantic embeddings (optional).
    description: A description of the training process (optional).
  Returns:
    None
  """
  import torch

  global current_dataset

  result_df = pd.DataFrame(columns=[
      'dataset', 'model', 'avg_loss', 'avg_test_loss', 'avg_val_loss', 'precision', 'recall', 'f1', 'accuracy'
  ])

  # read results from CSV file if it exists
  if os.path.exists('triplet_training_results.csv'):
    result_df = pd.read_csv('triplet_training_results.csv')

  print('##' * 20)
  # skip if the model is already trained and has results
  is_trained = False
  if (not result_df.empty) and (result_df[
      (result_df['model'] == model._get_name()) & (result_df['dataset'] == (f"{dataset_type}: {description}"))
  ].values.any()):
    print(f'Model {model._get_name()} on dataset {current_dataset} with description "{description}" is already trained.')
    # load the model and return
    model_path = ''
    if process_type == 'triplet':
      model_path = f'models/triplet/{current_dataset}_{model._get_name()}/model_fold_5.pth'
    elif process_type == 'classification':
      model_path = f'models/classification/{current_dataset}_{model._get_name()}/model_fold_5.pth'
    elif process_type == 'semantic_classification':
      model_path = f'models/classification/{current_dataset}_{model._get_name()}/model_fold_5.pth'
    elif process_type == 'AL_RL_semantic_classification':
      model_path = f'models/classification/{current_dataset}_{model._get_name()}/model_fold_5.pth'
    else:
      raise ValueError(f'Unknown process type: {process_type}')

    # Check if the model file exists
    if os.path.exists(model_path):
      print(f'Loading model from {model_path}...')
      model.load_state_dict(torch.load(model_path, weights_only=False))
      is_trained = True
    else:
      print(f'Model file {model_path} does not exist. Training again...')
      # remove the entry from the result_df
      result_df = result_df[~((result_df['model'] == model._get_name()) & (
          result_df['dataset'] == f"{dataset_type}: {description}"))]
      print(
          f'Results for model {model._get_name()} on dataset {dataset_type} with description "{description}" removed from results DataFrame.')
      # save the updated results DataFrame
      result_df.to_csv('triplet_training_results.csv', index=False)

  if not is_trained:
    print(f'Training {description} model {model._get_name()} on {train_data.__class__.__name__}...')
    trained_model, avg_loss, avg_val_loss, label_embedding = train(
        train_data, model, train_process=process_type,
        semantic_embedding_fn=semantic_fn,
        k_fold=5, batch_size=batch_size,
        test_dataset=test_data
    )
    print(f'Training completed for {description} model.')
  else:
    print(f'Loading pre-trained model for {description} model {model._get_name()}...')
    trained_model = model
    avg_loss, avg_val_loss, label_embedding = 0.0, 0.0, None
    # calculate label_embedding if process_type is semantic_classification
    if process_type == 'semantic_classification' or process_type == 'AL_RL_semantic_classification':
      label_embedding = compute_label_embeddings(
          train_data.labels, trained_model.backbone_out_features)
      print(f'Label embeddings computed for {description} model.')
      if not semantic_fn:
        # Default to zero if no function is provided, it means no semantic embeddings are used
        def semantic_fn(x): return 0
      label_embedding = {
          label: semantic_fn(index) * embedding for index, (label, embedding) in enumerate(label_embedding.items())
      }

  # Validate model on test set, except for triplet models
  if process_type == 'triplet':
    avg_test_loss, accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    print(f'Triplet model does not support test loss and accuracy calculation.')
  elif process_type == 'classification':
    avg_test_loss, accuracy, precision, recall, f1 = validate_model(trained_model, test_data, batch_size=batch_size)
    print(f'Results: Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, '
          f'Recall: {recall:.2f}%, F1 Score: {f1:.2f}%')
  elif process_type == 'semantic_classification' or process_type == 'AL_RL_semantic_classification':
    avg_test_loss, accuracy, precision, recall, f1 = semantic_validate_model(
        trained_model, test_data, train_data, label_to_embeddings=label_embedding, batch_size=batch_size)
    print(f'Results: Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, '
          f'Recall: {recall:.2f}%, F1 Score: {f1:.2f}%')
  else:
    raise ValueError(f'Unknown process type: {process_type}')

  # Record results
  result_df = pd.concat([result_df, pd.DataFrame({
      'model': [model._get_name()],
      'dataset': [f"{dataset_type}: {description}"],
      'avg_loss': [avg_loss],
      'avg_val_loss': [avg_val_loss],
      'avg_test_loss': [avg_test_loss],
      'accuracy': [accuracy],
      'precision': [precision],
      'recall': [recall],
      'f1': [f1]
  })], ignore_index=True)
  # result_df = result_df.sort_values(by=['dataset', 'model']).reset_index(drop=True)

  # Save results to CSV
  result_df.to_csv('triplet_training_results.csv', index=False)
  print(f'Results saved to triplet_training_results.csv')

  return trained_model, label_embedding


if __name__ == '__main__':
  import pandas as pd
  import dataset as ds
  import torchvision
  from model import FeatureExtractor, Classifier

  triplet_df = pd.DataFrame(columns=[
      'backbone_model',
      'feature_extractor_model',
      'dataset_type',
      'create_triplet_dataset_fn',
      'create_classification_dataset_fn',
      'create_classification_test_dataset_fn',
      'batch_size'
  ])
  classifier_df = pd.DataFrame(columns=['dataset', 'model', 'transform'])

  # gi4e_full dataset
  def create_gi4e_triplet_dataset_fn(): return ds.TripletGi4eDataset(
      './datasets/gi4e',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]))
  def create_gi4e_classification_dataset_fn(): return ds.Gi4eDataset(
      './datasets/gi4e',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
      is_classification=True)
  # Add training process for gi4e_full dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'gi4e_full',
      create_gi4e_triplet_dataset_fn,
      create_gi4e_classification_dataset_fn
  )], ignore_index=True)

  # gi4e_raw_eyes dataset
  def create_gi4e_raw_eyes_triplet_dataset_fn(): return ds.TripletImageDataset(
      './datasets/gi4e_raw_eyes',
      file_extension='png',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]))

  def create_gi4e_raw_eyes_classification_dataset_fn(): return ds.ImageDataset(
      './datasets/gi4e_raw_eyes',
      file_extension='png',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]))
  # Add training process for gi4e_raw_eyes dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'gi4e_raw_eyes',
      create_gi4e_raw_eyes_triplet_dataset_fn,
      create_gi4e_raw_eyes_classification_dataset_fn,
      models=['resnet'],
  )], ignore_index=True)

  # Youtube Faces dataset
  def create_youtube_faces_triplet_dataset_fn(): return ds.TripletYoutubeFacesDataset(
      data_path='./datasets/YouTubeFacesWithFacialKeypoints',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
      number_of_samples=10,  # Limit the number of samples for faster training
  )
  def create_youtube_faces_classification_dataset_fn(): return ds.YoutubeFacesWithFacialKeypoints(
      data_path='./datasets/YouTubeFacesWithFacialKeypoints',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.ToPILImage(),
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
      number_of_samples=10,  # Limit the number of samples for faster training
      is_classification=True
  )
  # Add training process for YouTube Faces dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'youtube_faces',
      create_youtube_faces_triplet_dataset_fn,
      create_youtube_faces_classification_dataset_fn,
      batch_size=16  # Adjust batch size as needed
  )], ignore_index=True)

  # CelebA dataset
  def create_celeb_a_triplet_dataset_fn(): return ds.TripletImageDataset(
      './datasets/CelebA_HQ_facial_identity_dataset/train',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
  )
  def create_celeb_a_classification_dataset_fn(): return ds.ImageDataset(
      './datasets/CelebA_HQ_facial_identity_dataset/train',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
  )
  def create_celeb_a_classification_test_dataset_fn(): return ds.ImageDataset(
      './datasets/CelebA_HQ_facial_identity_dataset/test',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
  )
  # Add training process for CelebA dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'celeb_a',
      create_celeb_a_triplet_dataset_fn,
      create_celeb_a_classification_dataset_fn,
      create_celeb_a_classification_test_dataset_fn
  )], ignore_index=True)

  # Nus2Hands dataset
  def create_nus2hands_triplet_dataset_fn(): return ds.TripletImageDataset(
      './datasets/nus2hands',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
  )
  def create_nus2hands_classification_dataset_fn(): return ds.ImageDataset(
      './datasets/nus2hands',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]),
  )
  # Add training process for Nus2Hands dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'nus2hands',
      create_nus2hands_triplet_dataset_fn,
      create_nus2hands_classification_dataset_fn,
      batch_size=32  # Adjust batch size as needed
  )], ignore_index=True)

  # FER2013 dataset
  input_size = (48, 48)  # FER2013 input size

  def create_fer2013_triplet_dataset_fn(): return ds.TripletImageDataset(
      './datasets/PER-2013/train',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize(input_size),
          torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
          torchvision.transforms.ToTensor()
      ]),
  )

  def create_fer2013_classification_dataset_fn(): return ds.ImageDataset(
      './datasets/PER-2013/train',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize(input_size),
          torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
          torchvision.transforms.ToTensor()
      ]),
  )

  def create_fer2013_classification_test_dataset_fn(): return ds.ImageDataset(
      './datasets/PER-2013/test',
      file_extension='jpg',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize(input_size),
          torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
          torchvision.transforms.ToTensor()
      ])
  )
  # Add training process for FER2013 dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'per2013',
      create_fer2013_triplet_dataset_fn,
      create_fer2013_classification_dataset_fn,
      # create_fer2013_classification_test_dataset_fn,
      models=['vgg'],
      batch_size=256  # Adjust batch size as needed
  )], ignore_index=True)

  # The process of training models following the triplet loss approach the same way as classification
  # loop through datasets and train triplet models
  for index, row in triplet_df.iterrows():
    batch_size = row['batch_size']
    backbone_model = row['backbone_model']
    triplet_model = row['feature_extractor_model']
    dataset_type = row['dataset_type']
    create_gi4e_triplet_dataset_fn = row['create_triplet_dataset_fn']
    create_classification_dataset_fn = row['create_classification_dataset_fn']
    create_classification_test_dataset_fn = row.get(
        'create_classification_test_dataset_fn', None)
    current_backbone_model = backbone_model

    # Run Cross-Validation on dataset to verify raw performance of backbone model
    print(f'Running Cross-Validation on dataset {dataset_type} with model {backbone_model._get_name()}...')

    # Create datasets
    train_ds, test_ds = create_train_test_dataset(
        create_classification_dataset_fn,
        create_classification_test_dataset_fn
    )
    classifier_dataset = create_classification_dataset_fn()
    triplet_dataset = create_gi4e_triplet_dataset_fn()

    # # 1. Raw backbone model evaluation
    # current_dataset = f"{dataset_type}_Cross-Validation_None"
    # trained_backbone, _ = train_and_evaluate(
    #     backbone_model, train_ds, test_ds, 'classification',
    #     description="Cross-Validation"
    # )

    # 2. Train triplet model
    print(f'Training triplet model on {triplet_dataset.__class__.__name__}...')
    current_dataset = f"{dataset_type}_Triplet"
    trained_triplet, _ = train_and_evaluate(
        triplet_model, triplet_dataset, None, 'triplet',
        description="Triplet"
    )
    # 3. Create embedded datasets using the trained triplet model
    print(f'Creating embedded datasets for {dataset_type} using trained triplet model...')
    # Create embedded datasets with the trained triplet model
    embedded_train_ds = ds.EmbeddedDataset(train_ds, trained_triplet, is_moving_labels_to_function=False)
    embedded_test_ds = ds.EmbeddedDataset(test_ds, trained_triplet, is_moving_labels_to_function=False)
    print(f'Embedded datasets created for {dataset_type}.')

    # # 4. Train classifier with triplet embeddings (no label moving)
    # current_dataset = f"{dataset_type}_Cross-Validation_Triplet"
    # trained_classifier, _ = train_and_evaluate(
    #     Classifier(trained_triplet), embedded_train_ds, embedded_test_ds, 'semantic_classification',
    #     semantic_fn=None, description="Cross-Validation + Triplet"
    # )

    # # 5. Train classifier with x-to-x label moving
    # current_dataset = f"{dataset_type}_Cross-Validation_Triplet_Moving_Labels_x-to-x"
    # trained_classifier_x_to_x, _ = train_and_evaluate(
    #     Classifier(trained_triplet), embedded_train_ds, embedded_test_ds, 'semantic_classification',
    #     semantic_fn=lambda x: x, description="Cross-Validation + Triplet + Moving Labels - x => x"
    # )

    # # 6. Train classifier with x-to-4x label moving
    # current_dataset = f"{dataset_type}_Cross-Validation_Triplet_Moving_Labels_x-to-4x"
    # trained_classifier_x_to_4x, _ = train_and_evaluate(
    #     Classifier(trained_triplet), embedded_train_ds, embedded_test_ds, 'semantic_classification',
    #     semantic_fn=lambda x: 4 * x, description="Cross-Validation + Triplet + Moving Labels - x => 4*x"
    # )

    # 7. Train classifier with x-to-4x label moving with AL-RL
    current_dataset = f"{dataset_type}_Cross-Validation_Triplet_Moving_Labels_AL_RL"
    trained_classifier_al_rl_x_to_4x, _ = train_and_evaluate(
        Classifier(trained_triplet), embedded_train_ds, embedded_test_ds, 'AL_RL_semantic_classification',
        semantic_fn=lambda x: 4 * x, description="Cross-Validation + Triplet + AL-RL Moving Labels - x => 4*x"
    )

  print('Training process completed.')
