import cv2
import numpy as np

def generate_chessboard(square_size=50, grid_size=(9, 6), output_file="chessboard.png"):
    board_width, board_height = grid_size
    image_size = (board_width * square_size, board_height * square_size)
    
    # Create the chessboard pattern
    board = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    for row in range(board_height):
        for col in range(board_width):
            if (row + col) % 2 == 0:
                x = col * square_size
                y = row * square_size
                board[y:y+square_size, x:x+square_size] = 255

    cv2.imwrite(output_file, board)
    print(f"Chessboard image saved as {output_file}")

# Example usage
generate_chessboard(grid_size=(9, 6))