import os
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from collections import deque
import pygame
import time
import math
import pickle
from tqdm import tqdm
import torch.nn.functional as F

# --- Configuration ---
ENV_ID = 'LunarLander-v3'
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAJECTORY_FILE = 'datasets/trajectories_5000.pkl'
MIN_EXPERT_RETURN = 250

# Optimized Decision Transformer Hyperparameters
DT_CONTEXT_LEN = 20
DT_N_HEADS = 8  # Reduced from 8
DT_N_LAYERS = 6  # Reduced from 6
DT_EMBED_DIM = 256
DT_DROPOUT = 0.1
DT_LR = 3e-4  # Increased learning rate
DT_WEIGHT_DECAY = 1e-4
DT_BATCH_SIZE = 256  # Optimal batch size
DT_NUM_EPOCHS = 50  # Reduced epochs with better learning
DT_TRAJECTORIES_TO_USE = 5000

# Enhanced Early Stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Learning Rate Scheduling
USE_LR_SCHEDULER = True
SCHEDULER_FACTOR = 0.8
SCHEDULER_PATIENCE = 3

# Data Augmentation
USE_DATA_AUGMENTATION = True
AUGMENTATION_NOISE_STD = 0.01

# Pygame Visualization
PYGAME_FPS = 60
NUM_EVAL_EPISODES = 10

# Global Normalization Parameters
state_mean = None
state_std = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def load_trajectories_from_pickle(file_path):
    print(f"\n--- Loading Expert Trajectories from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: Trajectory file not found at {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"Successfully loaded {len(trajectories)} trajectories.")
        return trajectories
    except Exception as e:
        print(f"Error loading trajectories from pickle file: {e}")
        return None


# Normalizing states helps the model train better by:
# Centering inputs around zero mean
# Scaling them to similar magnitudes
# This is especially important in models like Decision Transformer (DT) or any neural net, because unnormalized input can cause:
# Poor gradient flow
# Slower convergence
# Overfitting to high-magnitude features


def calculate_state_normalization_params(trajectories):
    """Efficiently calculates normalization parameters using vectorized operations."""
    print("Calculating state normalization parameters...")
    
    # Collect all states more efficiently
    all_states = []
    for traj in trajectories:
        all_states.append(np.array(traj['states']))
    
    # Concatenate all at once
    all_states = np.concatenate(all_states, axis=0).astype(np.float32)
    
    mean = np.mean(all_states, axis=0, keepdims=True)
    std = np.std(all_states, axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)  # Prevent division by zero
    
    print(f"Processed {len(all_states)} states for normalization")
    return mean.flatten(), std.flatten()

# normal (x-mu)/sigma
def normalize_state(state, mean, std):
    """Vectorized state normalization."""
    return (state - mean) / std

def process_trajectories_optimized(raw_trajectories, context_len, gamma=0.995, state_mean=None, state_std=None):
    """Optimized trajectory processing with better sampling strategy."""
    processed_data = []
    
    print("Processing trajectories with optimized sampling...")
    
    for trajectory in tqdm(raw_trajectories, desc="Processing trajectories"):
        states = np.array(trajectory['states'], dtype=np.float32)
        actions = np.array(trajectory['actions'], dtype=np.int64)
        rewards = np.array(trajectory['rewards'], dtype=np.float32)
        
        # Vectorized normalization
        normalized_states = (states - state_mean) / state_std
        
        # Vectorized returns-to-go calculation
        returns_to_go = np.zeros_like(rewards)
        returns_to_go[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            returns_to_go[t] = rewards[t] + gamma * returns_to_go[t + 1]
        
        traj_len = len(normalized_states)
        
        # Smart sampling: sample more from high-value parts of trajectory
        high_value_threshold = np.percentile(returns_to_go, 70)
        high_value_indices = np.where(returns_to_go >= high_value_threshold)[0]
        
        # Sample from high-value states more frequently
        sample_indices = []
        for i in range(0, traj_len, max(1, traj_len // 20)):  # Base sampling
            sample_indices.append(i)
        
        # Add extra samples from high-value states
        for idx in high_value_indices[::2]:  # Every other high-value state
            if idx not in sample_indices:
                sample_indices.append(idx)
        
        sample_indices = sorted(set(sample_indices))
        
        for i in sample_indices:
            end_idx = i + 1
            start_idx = max(0, end_idx - context_len)
            
            # Extract sequences
            s_seq = normalized_states[start_idx:end_idx]
            a_seq = actions[start_idx:end_idx]
            r_seq = returns_to_go[start_idx:end_idx]
            timesteps = np.arange(len(s_seq))
            
            # Pad sequences
            pad_len = context_len - len(s_seq)
            if pad_len > 0:
                s_seq = np.vstack([np.zeros((pad_len, s_seq.shape[1])), s_seq])
                a_seq = np.concatenate([np.zeros(pad_len, dtype=np.int64), a_seq])
                r_seq = np.concatenate([np.zeros(pad_len), r_seq])
                timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps + pad_len])
            
            processed_data.append({
                'states': s_seq.astype(np.float32),
                'actions': a_seq.astype(np.int64),
                'returns_to_go': r_seq.astype(np.float32),
                'timesteps': timesteps.astype(np.int64),
                'target_action': actions[i]
            })
    
    print(f"Generated {len(processed_data)} training samples")
    return processed_data

class OptimizedTrajectoryDataset(Dataset):
    def __init__(self, data, use_augmentation=False, noise_std=0.01):
        self.data = data
        self.use_augmentation = use_augmentation
        self.noise_std = noise_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        states = item['states'].copy()
        
        # Data augmentation: add small noise to states
        if self.use_augmentation and random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, states.shape).astype(np.float32)
            states = states + noise
        
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(item['actions'], dtype=torch.long),
                torch.tensor(item['returns_to_go'], dtype=torch.float32),
                torch.tensor(item['timesteps'], dtype=torch.long),
                torch.tensor(item['target_action'], dtype=torch.long))
    


# the dimension here for the embeddings used for it
class OptimizedDecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, context_len, n_heads, n_layers, dropout, max_timestep=4096):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.context_len = context_len

        #  it first loads the dimension of the state vector representation and then uses a neural network to get a standard embedding to embed_dim

        # More efficient embeddings
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # no need to use a neural network here as actions are categorical so nn.Embedding is much more efficient for embedding
        # it makes it into a row vector of (embed_dim,)
        self.action_embedding = nn.Embedding(action_dim, embed_dim)


        self.return_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Learnable positional embedding
        # 3*context_len as we have states, actions and returns per timestep or sequence
        # multiplied by 0.02 for decreasing very large number
        # nn.Parameter is used to create a learnable parameter which is optimized during training
        self.pos_embedding = nn.Parameter(torch.randn(1, context_len * 3, embed_dim) * 0.02)
        
        # More efficient transformer
        # architecture with pre-norm and reduced feedforward dimension
        # Pre-norm helps with training stability
        self.dropout = nn.Dropout(dropout)
        
        # Use more efficient transformer architecture
        # a single layer of the transformer encoder with pre-norm
        # and reduced feedforward dimension (2x instead of 4x)
        # This helps with training speed and stability
        # d_model is the embedding dimension
        # nhead is the number of attention heads
        # dim_feedforward is the dimension of the feedforward network
        # dropout is the dropout rate
        # activation is the activation function used in the feedforward network
        # batch_first=True means the input and output tensors are of shape (batch, seq, feature)
        # norm_first=False means the normalization is applied after the attention and feedforward layers
        # This is a more efficient transformer encoder layer
        # d_k= embed_dim // n_heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,  
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  
        )

        # Stack multiple layers of the transformer encoder
        # This allows the model to learn more complex representations
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Action prediction head with better architecture
        # it takes the output of the transformer and predicts the next action (discrete to categorical)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    # Initialize weights for the model
    # Xavier initialization for linear layers, normal for embeddings, zeros for biases
    # and ones for layer norms
    # it is random initialization of the weights of the model but in a way that helps the model to converge faster using statistics
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    # Forward pass of the model
    # it takes the states, actions, returns_to_go and timesteps as input
    # it embeds each modality (state, action, return_to_go)
    # then stacks them efficiently into a single tensor
    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embed each modality (states and returns_to_go use the neural network embedding whereas actions use nn.Embedding)
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        return_embeddings = self.return_embedding(returns_to_go.unsqueeze(-1))
        # what does unsqueeze(-1) do?
        # it adds a new dimension at the end of the tensor, making it (batch_size, seq_len, 1)  
        # this is useful for the linear layer to work with the input shape (batch_size, seq_len, 1) -> (batch_size, seq_len, embed_dim)
        # it is used to make the input shape compatible with the linear layer
        
        # Stack tokens efficiently as (batch_size, seq_len, 3, embed_dim)
        # This creates a tensor of shape (batch_size, seq_len, 3, embed_dim)
        # where 3 corresponds to return, state and action embeddings
        # then reshape it to (batch_size, seq_len * 3, embed_dim)
        # This is more efficient than concatenating and allows the transformer to process each modality separately
        stacked_inputs = torch.stack([return_embeddings, state_embeddings, action_embeddings], dim=2)
        stacked_inputs = stacked_inputs.reshape(batch_size, seq_len * 3, self.embed_dim)
        # basically, each element was a tuple of (return, state, action) embeddings so we split each element into 3 to get the sequence we need for training our decision transformer
        # stacked_inputs is now of shape (batch_size, seq_len * 3, embed_dim)
        
        # Add positional embeddings
        # the dimensions are matched to the stacked inputs automatically by broadcasting
        # this is a learnable parameter that helps the model to understand the order of the tokens in the sequence
        stacked_inputs = stacked_inputs + self.pos_embedding[:, :seq_len * 3, :]
        stacked_inputs = self.dropout(stacked_inputs)
        
        # Optimized causal mask to prevent seeing future tokens
        # This mask is created to ensure that the model only attends to past and current tokens
        mask = self._create_optimized_causal_mask(seq_len, stacked_inputs.device)
        
        # Transformer forward pass
        # it processes the stacked inputs with the transformer encoder
        # the mask is applied to prevent the model from attending to future tokens
        transformer_outputs = self.transformer(stacked_inputs, mask=mask)
        
        # Extract state tokens and predict actions
        # Extract every third token starting from the second one (state tokens)
        # This is because the input sequence is structured as (return, state, action) for each timestep

        # start at index one and jump by 3 to get only the state tokens
        # this is because the input sequence is structured as (return, state, action) for each timestep
        state_tokens = transformer_outputs[:, 1::3, :]
        # we already have the state tokens in the transformer outputs, so we can directly use them to predict actions using the embedding
        # Predict actions using the action head
        action_preds = self.action_head(state_tokens)
        
        return action_preds

    def _create_optimized_causal_mask(self, seq_len, device):
        """Optimized causal mask creation."""
        total_len = seq_len * 3
        
        # Use torch operations for efficiency
        # base causal mask to prevent seeing future tokens
        # torch.triu creates an upper triangular matrix with -inf above the diagonal 
        mask = torch.triu(torch.ones(total_len, total_len, device=device) * float('-inf'), diagonal=1)
        
        # Adjust mask for Decision Transformer token structure
        for i in range(total_len):
            # get the specific sequence
            timestep = i // 3
            # gives the token type state, action, return
            token_type = i % 3
            
            if token_type == 1:  # state token
                mask[i, timestep * 3] = 0  # can see RTG
            elif token_type == 2:  # action token
                mask[i, timestep * 3:timestep * 3 + 2] = 0  # can see RTG and state
        
        return mask

def train_optimized_decision_transformer(model, dataloader, optimizer, scheduler, num_epochs, patience, min_delta):
    print("\n--- Training Optimized Decision Transformer ---")
    # puts model in training mode
    model.train()
    # Initialize early stopping parameters
    # best_loss keeps track of the best loss so far
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Mixed precision training for faster computation
    # amp - Automatic Mixed Precision is a PyTorch feature that allows you to use mixed precision training which is faster and uses less memory by using half-precision (float16) for some operations thus ensuring more memory is available for the model
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # path to save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/dummy.pkl'

    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        # all of these are provided by the dataloader
        for states, actions, returns_to_go, timesteps, target_actions in progress_bar:
            states = states.to(DEVICE, non_blocking=True)
            actions = actions.to(DEVICE, non_blocking=True)
            returns_to_go = returns_to_go.to(DEVICE, non_blocking=True)
            timesteps = timesteps.to(DEVICE, non_blocking=True)
            target_actions = target_actions.to(DEVICE, non_blocking=True)

            # Zero gradients before starting the backward pass
            optimizer.zero_grad()
            
            # Mixed precision forward pass using autocast
            # autocast is used to enable mixed precision training which is faster and uses less memory
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    action_preds = model(states, actions, returns_to_go, timesteps)
                    # use cross_entropy loss for action prediction
                    # Calculate loss
                    # only use the last action prediction for the loss calculation
                    loss = F.cross_entropy(action_preds[:, -1, :], target_actions)
                
                # scales the loss to prevent underflow in gradients before backward pass
                # this is temporary and does not change the loss value
                scaler.scale(loss).backward()
                # the .backward() performs backpropagation to compute gradients

                # now that backpropogation is done, unscale them to use them in the optimizer
                # unscale gradients and clip them to prevent exploding gradients
                # optimzer is the object that is actually responsible for updating the model parameters
                scaler.unscale_(optimizer)

                # the norm here is vector norm (square root of sum of squares of all gradients)
                # if the norm exceeds the threshold, it scales down the gradients by the same amount 
                # this is done to prevent exploding gradients which can cause the model to diverge
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # checks if the gradients are finite and exist (not Nan or Inf) and then applies optimizer. It is basically extra checks
                scaler.step(optimizer)
                # this one is permanent and updates the model parameters based on the gradients
                # updates the scaler for the next iteration based on the gradients underflowing or overflowing
                scaler.update()
            else:
                # no scaling here as everything is float32
                action_preds = model(states, actions, returns_to_go, timesteps)
                loss = F.cross_entropy(action_preds[:, -1, :], target_actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            # Calculate accuracy

            # we only care about the last action prediction for accuracy
            # this is because we are predicting the next action based on the current state and context
            # torch.no_grad() is used to disable gradient calculation for inference
            # this is useful for evaluation and inference as it saves memory and computation
            with torch.no_grad():
                # choose the action with the highest predicted probability
                # argmax returns the index of the maximum value along the specified dimension
                predicted_actions = torch.argmax(action_preds[:, -1, :], dim=-1)
                # this is simply the fraction of correct predictions
                # the .float() converts the boolean tensor to [1.0,0.0,...] format
                # mean() calculates the average of the tensor which is the accuracy
                accuracy = (predicted_actions == target_actions).float().mean()
            
            # Update progress bar
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss/num_batches:.4f}', 
                'Acc': f'{total_accuracy/num_batches:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Calculate average loss and accuracy for the epoch
        # this is the average loss and accuracy for the epoch
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Update learning rate
        # if using learning rate scheduler, step it with the average loss, this is done faster convergence
        if scheduler is not None:
            scheduler.step(avg_loss)
        
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        # Enhanced early stopping
        # if the average loss is less than the best loss so far minus the minimum delta, we save the model state
        # this is to prevent overfitting and save the best model state
        # if this happens means we got an improvement in the model performance better than min_delta so save it
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
            # the state_dict() method returns a dictionary containing the model's parameters and buffers for example the weights of each layer etc
            # this is the best model state so far
            best_model_state = model.state_dict().copy()
            torch.save({
                'model_state_dict': best_model_state,
                'state_mean': state_mean,
                'state_std': state_std,
                'model_config': {
                    'state_dim': model.state_dim,
                    'action_dim': model.action_dim,
                    'embed_dim': model.embed_dim,
                    'context_len': model.context_len,
                    'n_heads': DT_N_HEADS,
                    'n_layers': DT_N_LAYERS,
                    'dropout': DT_DROPOUT
                }
            }, model_path)
            print(f"✓ New best model saved with loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping after {epoch + 1} epochs!")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✓ Model restored to best state (loss: {best_loss:.4f})")
    
    return model

def init_pygame(width, height):
    pygame.init()
    screen = pygame.display.set_mode((width, height + 50))
    pygame.display.set_caption("Optimized Lunar Lander Decision Transformer")
    font = pygame.font.Font(None, 36)
    return screen, font

def render_env_pygame(env, screen, font, current_score, episode_num, total_episodes):
    frame = env.render()
    if frame is not None:
        frame = np.transpose(frame, (1, 0, 2))
        pygame_surface = pygame.surfarray.make_surface(frame)
        screen.fill((0, 0, 0))
        screen.blit(pygame_surface, (0, 0))
    
    # Enhanced display
    score_text = font.render(f"Score: {current_score:.1f}", True, (255, 255, 255))
    episode_text = font.render(f"Episode: {episode_num}/{total_episodes}", True, (255, 255, 255))
    
    screen.blit(score_text, (10, screen.get_height() - 40))
    screen.blit(episode_text, (10, screen.get_height() - 80))
    pygame.display.flip()

if __name__ == "__main__":
    print(f"🚀 Device: {DEVICE}")
    print(f"🎯 Using {DT_TRAJECTORIES_TO_USE} trajectories with return >= {MIN_EXPERT_RETURN}")

    # Initialize environment
    env_dt = gym.make(ENV_ID, render_mode='rgb_array')
    state_dim = env_dt.observation_space.shape[0]
    action_dim = env_dt.action_space.n
    max_timestep = env_dt.spec.max_episode_steps if env_dt.spec else 1000

    # Load and filter trajectories
    raw_expert_trajectories = load_trajectories_from_pickle(TRAJECTORY_FILE)
    if not raw_expert_trajectories:
        print("❌ No trajectories loaded. Exiting.")
        exit()

    # Smart filtering: prioritize high-performing trajectories
    trajectory_returns = [(i, sum(traj['rewards'])) for i, traj in enumerate(raw_expert_trajectories)]
    trajectory_returns.sort(key=lambda x: x[1], reverse=True)  # Sort by return (descending)
    
    # Take top trajectories and some random ones for diversity
    top_indices = [idx for idx, ret in trajectory_returns[:int(DT_TRAJECTORIES_TO_USE * 0.8)] if ret >= MIN_EXPERT_RETURN]
    remaining_good = [idx for idx, ret in trajectory_returns[int(DT_TRAJECTORIES_TO_USE * 0.8):] if ret >= MIN_EXPERT_RETURN]
    
    if len(remaining_good) > 0:
        random_indices = random.sample(remaining_good, min(len(remaining_good), DT_TRAJECTORIES_TO_USE - len(top_indices)))
        selected_indices = top_indices + random_indices
    else:
        selected_indices = top_indices
    
    filtered_trajectories = [raw_expert_trajectories[i] for i in selected_indices[:DT_TRAJECTORIES_TO_USE]]
    
    print(f"✓ Selected {len(filtered_trajectories)} high-quality trajectories")
    print(f"  📊 Mean return: {np.mean([sum(traj['rewards']) for traj in filtered_trajectories]):.1f}")

    # Calculate normalization parameters
    state_mean, state_std = calculate_state_normalization_params(filtered_trajectories)

    # Process trajectories with optimization
    processed_dt_data = process_trajectories_optimized(
        filtered_trajectories, DT_CONTEXT_LEN, gamma=0.995, 
        state_mean=state_mean, state_std=state_std
    )
    
    dt_dataset = OptimizedTrajectoryDataset(processed_dt_data, USE_DATA_AUGMENTATION, AUGMENTATION_NOISE_STD)
    dt_dataloader = DataLoader(
        dt_dataset, 
        batch_size=DT_BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # Initialize optimized model
    dt_model = OptimizedDecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=DT_EMBED_DIM,
        context_len=DT_CONTEXT_LEN,
        n_heads=DT_N_HEADS,
        n_layers=DT_N_LAYERS,
        dropout=DT_DROPOUT,
        max_timestep=max_timestep
    ).to(DEVICE)

    # Optimized optimizer and scheduler
    dt_optimizer = optim.AdamW(
        dt_model.parameters(), 
        lr=DT_LR, 
        weight_decay=DT_WEIGHT_DECAY,
        betas=(0.9, 0.95)  # Better for transformers
    )
    
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            dt_optimizer, mode='min', factor=SCHEDULER_FACTOR, 
            patience=SCHEDULER_PATIENCE
        )

    print(f"🔧 Model parameters: {sum(p.numel() for p in dt_model.parameters()):,}")

    # Train model
    start_time = time.time()
    dt_model = train_optimized_decision_transformer(
        dt_model, dt_dataloader, dt_optimizer, scheduler, 
        DT_NUM_EPOCHS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
    )
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f} seconds")

    # Evaluation
    print("\n🎮 Starting evaluation...")
    input("Press Enter to continue...")
    
    try:
        NUM_EVAL_EPISODES = int(input("Enter number of episodes to watch: "))
    except:
        NUM_EVAL_EPISODES = 5

    # Initialize pygame
    _ = env_dt.reset()
    dummy_frame = env_dt.render()
    render_width = dummy_frame.shape[1] if dummy_frame is not None else 600
    render_height = dummy_frame.shape[0] if dummy_frame is not None else 400

    pygame_screen, pygame_font = init_pygame(render_width, render_height)
    clock = pygame.time.Clock()

    dt_model.eval()
    eval_returns = []
    
    for i_episode in range(NUM_EVAL_EPISODES):
        state, _ = env_dt.reset()
        state = normalize_state(state, state_mean, state_std)

        # Initialize context
        states = deque(maxlen=DT_CONTEXT_LEN)
        actions = deque(maxlen=DT_CONTEXT_LEN)
        returns_to_go = deque(maxlen=DT_CONTEXT_LEN)
        timesteps = deque(maxlen=DT_CONTEXT_LEN)

        current_episode_return = 0
        target_return = MIN_EXPERT_RETURN * 1.2  # Aim higher

        # Pad context
        for _ in range(DT_CONTEXT_LEN):
            states.append(np.zeros(state_dim, dtype=np.float32))
            actions.append(0)
            returns_to_go.append(0.0)
            timesteps.append(0)

        with torch.no_grad():
            for t in range(max_timestep):
                # Update context
                states.append(state)
                returns_to_go.append(max(target_return - current_episode_return, 0))
                timesteps.append(t)
                
                # Model inference
                s_input = torch.tensor(np.array(list(states)), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_input = torch.tensor(np.array(list(actions)), dtype=torch.long, device=DEVICE).unsqueeze(0)
                r_input = torch.tensor(np.array(list(returns_to_go)), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                t_input = torch.tensor(np.array(list(timesteps)), dtype=torch.long, device=DEVICE).unsqueeze(0)

                action_preds = dt_model(s_input, a_input, r_input, t_input)
                predicted_action = torch.argmax(action_preds[0, -1, :]).item()

                # Environment step
                next_state, reward, terminated, truncated, _ = env_dt.step(predicted_action)
                done = terminated or truncated
                
                next_state = normalize_state(next_state, state_mean, state_std)
                current_episode_return += reward
                state = next_state
                actions.append(predicted_action)

                # Render
                render_env_pygame(env_dt, pygame_screen, pygame_font, current_episode_return, i_episode + 1, NUM_EVAL_EPISODES)
                clock.tick(PYGAME_FPS)

                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                if done:
                    break

        eval_returns.append(current_episode_return)
        print(f"Episode {i_episode + 1}: Return = {current_episode_return:.2f}")

    print(f"\n📊 Evaluation Results:")
    print(f"Mean Return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    print(f"Best Return: {np.max(eval_returns):.2f}")
    print(f"Success Rate: {sum(1 for r in eval_returns if r >= 200)/len(eval_returns)*100:.1f}%")
    
    env_dt.close()
    pygame.quit()
    print("✅ Evaluation complete!")