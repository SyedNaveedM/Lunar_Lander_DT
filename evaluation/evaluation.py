import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pygame
from collections import deque
import pickle # Added for loading normalization params

# Configuration constants (matching your training setup)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT_CONTEXT_LEN = 20
# These will be loaded from the model checkpoint
STATE_DIM = 8
NUM_ACTIONS = 4

# These hyperparameters should ideally be loaded from model_config in the checkpoint
# but are set here as defaults in case model_config is not fully used by an older checkpoint
DT_EMBED_DIM = 256
DT_N_HEADS = 8
DT_N_LAYERS = 6
DT_DROPOUT = 0.1
MAX_TIMESTEP = 1000 # Max episode steps for LunarLander-v3 is typically 1000
MIN_EXPERT_RETURN = 300 # Align with training's MIN_EXPERT_RETURN
PYGAME_FPS = 60

# Decision Transformer Model Definition (exact copy from lunar_lander_2.py's OptimizedDecisionTransformer)
# Renamed to DecisionTransformer for consistency with the original eval script's main function
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, context_len, n_heads, n_layers, dropout, max_timestep=4096):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.context_len = context_len

        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.action_embedding = nn.Embedding(action_dim, embed_dim)

        self.return_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, context_len * 3, embed_dim) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=2 * embed_dim,  # Reduced from 4x
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, action_dim)
        )
        
        self.apply(self._init_weights)

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

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        return_embeddings = self.return_embedding(returns_to_go.unsqueeze(-1))
        
        stacked_inputs = torch.stack([return_embeddings, state_embeddings, action_embeddings], dim=2)
        stacked_inputs = stacked_inputs.reshape(batch_size, seq_len * 3, self.embed_dim)
        
        stacked_inputs = stacked_inputs + self.pos_embedding[:, :seq_len * 3, :]
        stacked_inputs = self.dropout(stacked_inputs)
        
        mask = self._create_optimized_causal_mask(seq_len, stacked_inputs.device)
        
        transformer_outputs = self.transformer(stacked_inputs, mask=mask)
        
        state_tokens = transformer_outputs[:, 1::3, :]
        action_preds = self.action_head(state_tokens)
        
        return action_preds

    def _create_optimized_causal_mask(self, seq_len, device):
        total_len = seq_len * 3
        
        mask = torch.triu(torch.ones(total_len, total_len, device=device) * float('-inf'), diagonal=1)
        
        for i in range(total_len):
            timestep = i // 3
            token_type = i % 3
            
            if token_type == 1:  # state token
                mask[i, timestep * 3] = 0  # can see RTG
            elif token_type == 2:  # action token
                mask[i, timestep * 3:timestep * 3 + 2] = 0  # can see RTG and state
        
        return mask

def normalize_state(state, mean, std):
    """Normalize state using provided mean and std"""
    return (state - mean) / (std + 1e-8) # Added epsilon to prevent division by zero

def init_pygame(width, height):
    """Initialize pygame for rendering"""
    pygame.init()
    # Add extra height for score/episode display, similar to training script's eval
    screen = pygame.display.set_mode((width, height + 50)) 
    pygame.display.set_caption("Optimized Lunar Lander Decision Transformer Evaluation")
    font = pygame.font.Font(None, 36)
    return screen, font

def render_env_pygame(env, screen, font, current_score, episode_num, total_episodes):
    """Render environment using pygame with score and episode display"""
    frame = env.render()
    if frame is not None:
        frame = np.transpose(frame, (1, 0, 2)) # Adjust for pygame surface
        pygame_surface = pygame.surfarray.make_surface(frame)
        screen.fill((0, 0, 0))
        screen.blit(pygame_surface, (0, 0))
    
    # Enhanced display (copied from training script's eval)
    score_text = font.render(f"Score: {current_score:.1f}", True, (255, 255, 255))
    episode_text = font.render(f"Episode: {episode_num}/{total_episodes}", True, (255, 255, 255))
    
    screen.blit(score_text, (10, screen.get_height() - 40))
    screen.blit(episode_text, (10, screen.get_height() - 80))
    pygame.display.flip()

# Main Evaluation Script
def main():
    print("\n--- Loading and Evaluating Decision Transformer ---")
    
    # Initialize environment
    env_dt = gym.make('LunarLander-v3', render_mode='rgb_array')
    # State and action dimensions should be derived from the environment
    state_dim = env_dt.observation_space.shape[0]
    action_dim = env_dt.action_space.n
    
    # Load the trained model and associated parameters
    model_path = '../models/lunar_lander_model_5000.pkl' 

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have trained and saved the model first using lunar_lander_2.py.")
        return
    
    print(f"Loading model and normalization parameters from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Extract config, state_mean, state_std from checkpoint
    model_config = checkpoint.get('model_config', { # Use .get() for robustness
        'state_dim': state_dim,
        'action_dim': action_dim,
        'embed_dim': DT_EMBED_DIM,
        'context_len': DT_CONTEXT_LEN,
        'n_heads': DT_N_HEADS,
        'n_layers': DT_N_LAYERS,
        'dropout': DT_DROPOUT
    })
    state_mean = checkpoint['state_mean']
    state_std = checkpoint['state_std']

    # Initialize model using loaded config
    dt_model = DecisionTransformer(
        state_dim=model_config['state_dim'],
        action_dim=model_config['action_dim'],
        embed_dim=model_config['embed_dim'],
        context_len=model_config['context_len'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        dropout=model_config['dropout'],
        max_timestep=MAX_TIMESTEP
    ).to(DEVICE)
    
    dt_model.load_state_dict(checkpoint["model_state_dict"])

    print("Model and normalization parameters loaded successfully!")
    
    input("Press Enter to continue to evaluation... ")
    
    try:
        NUM_EVAL_EPISODES = int(input("Enter number of episodes to watch: "))
    except ValueError:
        NUM_EVAL_EPISODES = 5 # Default if input is invalid or empty

    # Initialize pygame
    _ = env_dt.reset()
    dummy_frame = env_dt.render()
    render_width = dummy_frame.shape[1] if dummy_frame is not None else 600
    render_height = dummy_frame.shape[0] if dummy_frame is not None else 400
    
    pygame_screen, pygame_font = init_pygame(render_width, render_height)
    clock = pygame.time.Clock()
    
    dt_model.eval() # Set model to evaluation mode
    eval_returns = []
    
    print(f"\nRunning {NUM_EVAL_EPISODES} evaluation episodes...")
    
    for i_episode in range(NUM_EVAL_EPISODES):
        state, _ = env_dt.reset()
        state = normalize_state(state, state_mean, state_std) # Normalize initial state
    
        # Initialize context with padding
        states = deque(maxlen=DT_CONTEXT_LEN)
        actions = deque(maxlen=DT_CONTEXT_LEN)
        returns_to_go = deque(maxlen=DT_CONTEXT_LEN)
        timesteps = deque(maxlen=DT_CONTEXT_LEN)
    
        current_episode_return = 0
        target_return = MIN_EXPERT_RETURN * 1.2 # Align with training's eval target
    
        # Pad initially with zeros (copied from training script's eval)
        for _ in range(DT_CONTEXT_LEN):
            states.append(np.zeros(state_dim, dtype=np.float32))
            actions.append(0)
            returns_to_go.append(0.0)
            timesteps.append(0)
    
        with torch.no_grad():
            for t in range(MAX_TIMESTEP):
                # Update context
                states.append(state)
                returns_to_go.append(max(target_return - current_episode_return, 0)) # Max with 0
                timesteps.append(t)
                
                # Prepare model inputs 
                # Convert deques to numpy arrays then to tensors
                s_input = torch.tensor(np.array(list(states)), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a_input = torch.tensor(np.array(list(actions)), dtype=torch.long, device=DEVICE).unsqueeze(0)
                r_input = torch.tensor(np.array(list(returns_to_go)), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                t_input = torch.tensor(np.array(list(timesteps)), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
                # Get action prediction
                action_preds = dt_model(s_input, a_input, r_input, t_input)
                predicted_action = torch.argmax(action_preds[0, -1, :]).item() # Take last action
    
                # Take action
                next_state, reward, terminated, truncated, _ = env_dt.step(predicted_action)
                done = terminated or truncated
                
                next_state = normalize_state(next_state, state_mean, state_std) # Normalize next state
                current_episode_return += reward
                state = next_state
    
                # Update actions history
                actions.append(predicted_action)
    
                # Render
                render_env_pygame(env_dt, pygame_screen, pygame_font, current_episode_return, i_episode + 1, NUM_EVAL_EPISODES)
                clock.tick(PYGAME_FPS)
                
                # Handle pygame events (important for graceful exit)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env_dt.close()
                        exit()
    
                if done:
                    break
    
        eval_returns.append(current_episode_return)
        print(f"Episode {i_episode + 1}/{NUM_EVAL_EPISODES}: Return = {current_episode_return:.2f}")
    
    print(f"\n=== Evaluation Results ===")
    print(f"Episodes evaluated: {NUM_EVAL_EPISODES}")
    print(f"Mean Return: {np.mean(eval_returns):.2f} ± {np.std(eval_returns):.2f}")
    print(f"Min Return: {np.min(eval_returns):.2f}")
    print(f"Max Return: {np.max(eval_returns):.2f}")
    print(f"Success Rate (Return >= 200): {sum(1 for r in eval_returns if r >= 200)/len(eval_returns)*100:.1f}%") # Success rate added

    env_dt.close()
    pygame.quit()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()