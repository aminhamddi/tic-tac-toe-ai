import os
import random
import sys


import pygame
import numpy as np

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Please install: pip install tensorflow")
    TF_AVAILABLE = False

# Initialize PyGame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 700
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)


class DatasetLoader:
    """Load and process your existing Tic-Tac-Toe dataset"""

    def __init__(self):
        self.training_data = []
        self.games_loaded = 0

    def parse_dataset_line(self, line):
        """Parse a single line from your dataset format"""
        parts = line.strip().split(',')
        if len(parts) < 8:
            return None

        moves = []
        outcome = parts[-1].strip().lower()

        for move in parts[:7]:
            if move.strip() == '?':
                break
            try:
                moves.append(int(move.strip()))
            except ValueError:
                return None

        return moves, outcome

    def load_from_csv(self, filename="tictactoe_dataset.csv"):
        """Load dataset from CSV file"""
        if not os.path.exists(filename):
            print(f"Dataset file {filename} not found!")
            return False

        self.training_data = []
        self.games_loaded = 0

        with open(filename, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('MOVE'):
                    result = self.parse_dataset_line(line)
                    if result:
                        moves, outcome = result
                        self.training_data.append((moves, outcome))
                        self.games_loaded += 1

        print(f"Loaded {self.games_loaded} games from dataset")
        return True

    def create_training_examples(self, filter_losses=True):
        """Convert games to training examples - ONLY LEARN FROM GOOD MOVES"""
        if not self.training_data:
            print("No training data available!")
            return []

        training_examples = []
        wins_used = 0
        draws_used = 0
        losses_skipped = 0

        for game_moves, outcome in self.training_data:
            # CRITICAL FIX: Skip games where X lost - don't learn bad moves!
            if filter_losses and outcome == 'loss':
                losses_skipped += 1
                continue

            if outcome == 'win':
                wins_used += 1
            elif outcome == 'draw':
                draws_used += 1

            board = [[' ' for _ in range(3)] for _ in range(3)]
            current_player = 'X'

            # Only learn from moves in winning/drawing games
            for move in game_moves:
                row = move // 3
                col = move % 3

                # Create features from O's perspective (AI player)
                if current_player == 'O':
                    features = self.board_to_features(board, 'O')
                    print(features)
                    # Weight winning moves higher
                    weight = 1.5 if outcome == 'win' else 1.0
                    training_examples.append((features, move, weight))

                # Make the move
                board[row][col] = current_player
                current_player = 'O' if current_player == 'X' else 'X'

        print(f"Training data: {wins_used} wins, {draws_used} draws used")
        print(f"Skipped {losses_skipped} losing games")
        print(f"Created {len(training_examples)} training examples")
        return training_examples

    def board_to_features(self, board, current_player='O'):
        """Convert board to features from player's perspective"""
        features = []
        for row in range(3):
            for col in range(3):
                cell = board[row][col]
                if cell == current_player:
                    features.extend([1, 0, 0])  # My piece
                elif cell == ' ':
                    features.extend([0, 0, 1])  # Empty
                else:
                    features.extend([0, 1, 0])  # Opponent piece
        print(features)
        return features

    def generate_synthetic_data(self, num_games=1000):
        """Generate training data with strategic play"""
        print(f"Generating {num_games} synthetic games with strategy...")

        for _ in range(num_games):
            board = [[' ' for _ in range(3)] for _ in range(3)]
            moves = []
            current_player = 'X'

            while True:
                available_moves = []
                for row in range(3):
                    for col in range(3):
                        if board[row][col] == ' ':
                            available_moves.append(row * 3 + col)

                if not available_moves:
                    outcome = 'draw'
                    break

                # Use strategy for both players
                move = self.choose_strategic_move(board, available_moves, current_player)
                moves.append(move)

                row, col = move // 3, move % 3
                board[row][col] = current_player

                winner = self.check_winner(board)
                if winner:
                    outcome = 'win' if winner == 'X' else 'loss'
                    break

                current_player = 'O' if current_player == 'X' else 'X'

            self.training_data.append((moves, outcome))

        self.games_loaded += num_games
        print(f"Generated {num_games} synthetic games")

    def choose_strategic_move(self, board, available_moves, player):
        """Strategic move selection"""
        opponent = 'O' if player == 'X' else 'X'

        # 1. Win if possible
        for move in available_moves:
            row, col = move // 3, move % 3
            board[row][col] = player
            if self.check_winner(board) == player:
                board[row][col] = ' '
                return move
            board[row][col] = ' '

        # 2. Block opponent from winning
        for move in available_moves:
            row, col = move // 3, move % 3
            board[row][col] = opponent
            if self.check_winner(board) == opponent:
                board[row][col] = ' '
                return move
            board[row][col] = ' '

        # 3. Take center
        if 4 in available_moves:
            return 4

        # 4. Take corners
        corners = [0, 2, 6, 8]
        available_corners = [c for c in corners if c in available_moves]
        if available_corners:
            return random.choice(available_corners)

        # 5. Take any edge
        return random.choice(available_moves)

    def check_winner(self, board):
        """Check for winner"""
        # Rows
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2] != ' ':
                return board[row][0]

        # Columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != ' ':
                return board[0][col]

        # Diagonals
        if board[0][0] == board[1][1] == board[2][2] != ' ':
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != ' ':
            return board[0][2]

        return None


class SupervisedLearningAI:
    """AI trained only on good moves"""

    def __init__(self):
        self.model = None
        self.is_trained = False

    def build_model(self):
        if not TF_AVAILABLE:
            print("TensorFlow not available.")
            return False

        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(27,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(9, activation='softmax')
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Neural network built!")
        return True

    def train_model(self, training_examples, epochs=100):
        """Train on weighted examples"""
        if not TF_AVAILABLE or self.model is None:
            return False

        # Extract features, moves, and weights
        X = np.array([ex[0] for ex in training_examples])
        moves = [ex[1] for ex in training_examples]
        weights = np.array([ex[2] for ex in training_examples])

        # One-hot encode moves
        y = np.zeros((len(training_examples), 9))
        for i, move in enumerate(moves):
            y[i, move] = 1.0

        print(f"Training on {len(X)} examples for {epochs} epochs...")

        # Train with sample weights
        history = self.model.fit(
            X, y,
            sample_weight=weights,
            epochs=epochs,
            batch_size=32,
            validation_split=0.15,
            verbose=1
        )

        self.is_trained = True
        print("Training completed!")

        final_acc = history.history['accuracy'][-1]
        val_acc = history.history.get('val_accuracy', [0])[-1]
        print(f"Final Training Accuracy: {final_acc:.2%}")
        print(f"Final Validation Accuracy: {val_acc:.2%}")

        return True

    def predict_move(self, board, player='O'):
        """Predict best available move with strategy fallback"""
        if not self.is_trained or self.model is None:
            return None

        # Get available moves
        available_moves = []
        for row in range(3):
            for col in range(3):
                if board[row][col] == ' ':
                    available_moves.append(row * 3 + col)

        if not available_moves:
            return None

        # Check for immediate win
        win_move = self.check_winning_move(board, player, available_moves)
        if win_move is not None:
            return win_move

        # Check for blocking move
        opponent = 'X' if player == 'O' else 'O'
        block_move = self.check_winning_move(board, opponent, available_moves)
        if block_move is not None:
            return block_move

        # Use neural network for other moves
        dataset_loader = DatasetLoader()
        features = np.array([dataset_loader.board_to_features(board, player)])
        predictions = self.model.predict(features, verbose=0)[0]

        # Choose best available move from predictions
        best_move = None
        best_score = -1

        for move in available_moves:
            if predictions[move] > best_score:
                best_score = predictions[move]
                best_move = move

        return best_move

    def check_winning_move(self, board, player, available_moves):
        """Check if there's a winning move available"""
        for move in available_moves:
            row, col = move // 3, move % 3
            board[row][col] = player

            # Check if this creates a win
            winner = self.check_winner(board)
            board[row][col] = ' '

            if winner == player:
                return move
        return None

    def check_winner(self, board):
        """Check for winner"""
        # Rows
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2] != ' ':
                return board[row][0]

        # Columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != ' ':
                return board[0][col]

        # Diagonals
        if board[0][0] == board[1][1] == board[2][2] != ' ':
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != ' ':
            return board[0][2]

        return None

    def save_model(self, filename="supervised_model.keras"):
        """Save model"""
        if self.model and self.is_trained:
            self.model.save(filename)
            print(f"Model saved to {filename}")
            return True
        return False

    def load_model(self, filename="supervised_model.keras"):
        """Load model"""
        if not TF_AVAILABLE:
            return False

        # Try both .keras and .h5 extensions
        for ext in [filename, filename.replace('.keras', '.h5'), filename.replace('.h5', '.keras')]:
            if os.path.exists(ext):
                try:
                    self.model = keras.models.load_model(ext)
                    self.is_trained = True
                    print(f"Model loaded from {ext}")
                    return True
                except:
                    continue

        print(f"Model file not found")
        return False


class TicTacToeWithSupervisedAI:
    def __init__(self):
        self.reset_game()
        self.font = pygame.font.Font(None, 36)
        self.status_font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 24)

        self.dataset_loader = DatasetLoader()
        self.supervised_ai = SupervisedLearningAI()
        self.ai_mode = "supervised"
        self.training_in_progress = False

    def reset_game(self):
        """Reset game state"""
        self.board = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        self.moves_made = 0

    def draw_board(self):
        """Draw the game board"""
        screen.fill(WHITE)

        # Grid lines
        for i in range(1, BOARD_ROWS):
            pygame.draw.line(screen, BLACK, (0, i * SQUARE_SIZE), (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
            pygame.draw.line(screen, BLACK, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, WIDTH), LINE_WIDTH)

        # X's and O's
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 'X':
                    self.draw_x(row, col)
                elif self.board[row][col] == 'O':
                    self.draw_o(row, col)

        self.draw_status()

    def draw_x(self, row, col):
        """Draw X"""
        padding = SQUARE_SIZE // 4
        pygame.draw.line(
            screen, RED,
            (col * SQUARE_SIZE + padding, row * SQUARE_SIZE + padding),
            ((col + 1) * SQUARE_SIZE - padding, (row + 1) * SQUARE_SIZE - padding),
            LINE_WIDTH
        )
        pygame.draw.line(
            screen, RED,
            ((col + 1) * SQUARE_SIZE - padding, row * SQUARE_SIZE + padding),
            (col * SQUARE_SIZE + padding, (row + 1) * SQUARE_SIZE - padding),
            LINE_WIDTH
        )

    def draw_o(self, row, col):
        """Draw O"""
        padding = SQUARE_SIZE // 4
        radius = (SQUARE_SIZE - 2 * padding) // 2
        center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
        center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
        pygame.draw.circle(screen, BLUE, (center_x, center_y), radius, LINE_WIDTH)

    def draw_status(self):
        """Draw status area"""
        status_rect = pygame.Rect(0, WIDTH, WIDTH, HEIGHT - WIDTH)
        pygame.draw.rect(screen, GRAY, status_rect)

        y_offset = WIDTH + 20

        # Game status
        if self.game_over:
            if self.winner:
                text = f"Player {self.winner} wins!"
                color = GREEN
            else:
                text = "It's a tie!"
                color = BLACK
        else:
            if self.training_in_progress:
                text = "Training AI..."
                color = PURPLE
            elif self.current_player == 'X':
                text = "Your turn (X)"
                color = BLACK
            else:
                text = f"AI's turn (O) - {self.ai_mode}"
                color = ORANGE

        text_surface = self.status_font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, y_offset))
        screen.blit(text_surface, text_rect)
        y_offset += 40

        # AI status
        if self.supervised_ai.is_trained:
            ai_text = "Supervised AI: TRAINED"
            color = GREEN
        else:
            ai_text = "Supervised AI: Not trained"
            color = RED

        ai_surface = self.small_font.render(ai_text, True, color)
        ai_rect = ai_surface.get_rect(center=(WIDTH // 2, y_offset))
        screen.blit(ai_surface, ai_rect)
        y_offset += 30

        # Dataset info
        dataset_text = f"Games loaded: {self.dataset_loader.games_loaded}"
        dataset_surface = self.small_font.render(dataset_text, True, BLACK)
        dataset_rect = dataset_surface.get_rect(center=(WIDTH // 2, y_offset))
        screen.blit(dataset_surface, dataset_rect)
        y_offset += 30

        # Controls
        controls = [
            "R: Reset | L: Load Dataset | T: Train AI",
            "M: Load Model | S: Save Model",
            "G: Generate 1000 Games | Q: Quit",
            "1: Supervised AI | 2: Random AI"
        ]

        for control in controls:
            control_surface = self.small_font.render(control, True, BLACK)
            control_rect = control_surface.get_rect(center=(WIDTH // 2, y_offset))
            screen.blit(control_surface, control_rect)
            y_offset += 25

    def get_available_moves(self):
        """Get available moves"""
        available_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    available_moves.append((row, col))
        return available_moves

    def make_move(self, row, col):
        """Make a move"""
        if not self.game_over and self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.moves_made += 1

            self.winner = self.check_winner()
            if self.winner or self.is_board_full():
                self.game_over = True
                return True, self.winner
            else:
                self.switch_player()
                return True, None
        return False, None

    def check_winner(self):
        """Check for winner"""
        # Rows
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] != ' ':
                return self.board[row][0]

        # Columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]

        # Diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        return None

    def is_board_full(self):
        """Check if board is full"""
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    return False
        return True

    def switch_player(self):
        """Switch players"""
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def ai_make_move(self):
        """AI makes a move"""
        if self.game_over or self.current_player != 'O':
            return False

        if self.ai_mode == "supervised" and self.supervised_ai.is_trained:
            action = self.supervised_ai.predict_move(self.board, 'O')
            if action is not None:
                row = action // 3
                col = action % 3
                return self.make_move(row, col)

        # Fallback to random
        available_moves = self.get_available_moves()
        if available_moves:
            row, col = random.choice(available_moves)
            return self.make_move(row, col)

        return False

    def load_dataset(self):
        """Load dataset"""
        success = self.dataset_loader.load_from_csv("tictactoe_dataset.csv")
        if success:
            print(f"Loaded {self.dataset_loader.games_loaded} games")
        return success

    def train_supervised_ai(self):
        """Train AI"""
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return

        if not self.dataset_loader.training_data:
            print("No dataset loaded. Load dataset first (L key)")
            return

        print("Starting training...")
        self.training_in_progress = True

        if not self.supervised_ai.model:
            self.supervised_ai.build_model()

        # Create training examples (filter out losses!)
        training_examples = self.dataset_loader.create_training_examples(filter_losses=True)

        if not training_examples:
            print("No training examples created!")
            self.training_in_progress = False
            return

        success = self.supervised_ai.train_model(training_examples, epochs=100)

        self.training_in_progress = False

        if success:
            self.supervised_ai.save_model()
            print("Training completed and model saved!")
        else:
            print("Training failed!")

    def handle_click(self, pos):
        """Handle mouse click"""
        if pos[1] < WIDTH and not self.training_in_progress:
            col = pos[0] // SQUARE_SIZE
            row = pos[1] // SQUARE_SIZE
            return self.make_move(row, col)
        return False, None

    def run(self):
        """Main game loop"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_l:
                        self.load_dataset()
                    elif event.key == pygame.K_t and not self.training_in_progress:
                        self.train_supervised_ai()
                    elif event.key == pygame.K_m:
                        self.supervised_ai.load_model()
                    elif event.key == pygame.K_s:
                        self.supervised_ai.save_model()
                    elif event.key == pygame.K_1:
                        self.ai_mode = "supervised"
                        print("AI mode: Supervised Learning")
                    elif event.key == pygame.K_2:
                        self.ai_mode = "random"
                        print("AI mode: Random")
                    elif event.key == pygame.K_g:
                        self.dataset_loader.generate_synthetic_data(1000)
                    elif event.key == pygame.K_q:
                        running = False

            # AI move
            if not self.game_over and not self.training_in_progress and self.current_player == 'O':
                self.ai_make_move()
                pygame.time.delay(500)

            self.draw_board()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()


def create_sample_dataset():
    """Create sample dataset"""
    sample_data = """MOVE1,MOVE2,MOVE3,MOVE4,MOVE5,MOVE6,MOVE7,CLASS
0,8,1,3,?,?,?,loss
4,7,2,6,?,?,?,win
0,8,1,6,5,?,?,draw
0,4,1,2,6,?,?,win
3,0,4,1,5,?,?,loss
0,1,4,3,5,8,2,draw
2,0,1,4,7,5,8,win"""

    with open("tictactoe_dataset.csv", "w") as f:
        f.write(sample_data)
    print("Sample dataset created")


if __name__ == "__main__":
    if not os.path.exists("tictactoe_dataset.csv"):
        create_sample_dataset()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic-Tac-Toe with Supervised Learning AI")
    clock = pygame.time.Clock()

    game = TicTacToeWithSupervisedAI()
    game.run()