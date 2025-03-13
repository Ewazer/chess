import numpy as np

piece_note = {
    0: "0",
    1: "p",
    5: "t",
    4: "f",
    3: "c",
    9: "r",
    7: "k",
    -1: "P",
    -5: "T",
    -4: "F",
    -3: "C",
    -9: "R",
    -7: "K"
}

piece_note_style = {
    1: "♙",   
    5: "♖",   
    3: "♘",  
    4: "♗",   
    9: "♕",  
    7: "♔", 
    -1: "♟",
    -5: "♜",  
    -3: "♞", 
    -4: "♝",  
    -9: "♛", 
    -7: "♚",
    0: " "    
}

board = [
    [ 5, 0, 0, 0, 0, 4, 0, 5],
    [ 0, 1, 0, 0, 0, 0, 9, 0],
    [ 1, 0, 0, 7, 0, 3, 0, 1],
    [ 0, 0, 1, 1, 0, 0, 1, 0],
    [ 0, 0, 0, 0,-1,-1,-1, 3],
    [-3,-1,-1,-7,-1, 0, 0,-1],
    [ 0, 0,-5, 0, 0, 0, 0,-9],
    [-5, 0,-4, 0, 0, 0, 0, 0]
]

def board_print(style=True,color='white',board=board):
    if style:
        board_rendu = [list(reversed([piece_note_style[e] for e in r])) for r in board] if color == 'white' else [[piece_note_style[e] for e in r] for r in board]
    else:    
        board_rendu = [list(reversed([piece_note[e] for e in r])) for r in board]
    if color == 'white':
        board_rendu.reverse()
    for row in board_rendu:
        print(row)

WhitePawnTable = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 50,  50,  50,  50,  50,  50,  50,  50],
    [ 10,  10,  20,  30,  30,  20,  10,  10],
    [  5,   5,  10,  25,  25,  10,   5,   5],
    [  0,   0,   0,  20,  20,   0,   0,   0],
    [  5,  -5, -10,   0,   0, -10,  -5,   5],
    [  5,  10,  10, -20, -20,  10,  10,   5],
    [  0,   0,   0,   0,   0,   0,   0,   0]
], dtype=np.int32)

BlackPawnTable = WhitePawnTable[::-1]

WhiteKnightTable = np.array([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
], dtype=np.int32)

BlackKnightTable = WhiteKnightTable[::-1]

WhiteBishopTable = np.array([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
], dtype=np.int32)

BlackBishopTable = WhiteBishopTable[::-1]

WhiteRookTable = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0]
], dtype=np.int32)

BlackRookTable = WhiteRookTable[::-1]

WhiteQueenTable = np.array([
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20]
], dtype=np.int32)

BlackQueenTable = WhiteQueenTable[::-1]

WhiteKingTable = np.array([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20]
], dtype=np.int32)

BlackKingTable = WhiteKingTable[::-1]

PieceTable = {
    1:  WhitePawnTable,
    3:  WhiteKnightTable,
    4:  WhiteBishopTable,
    5:  WhiteRookTable,
    7:  WhiteKingTable,
    9:  WhiteQueenTable,
    -1: BlackPawnTable,
    -3: BlackKnightTable,
    -4: BlackBishopTable,
    -5: BlackRookTable,
    -7: BlackKingTable,
    -9: BlackQueenTable
}

piece_values = {
    1: 100,    # Pion
    3: 320,    # Cavalier
    4: 330,    # Fou
    5: 500,    # Tour
    9: 900,    # Dame
    7: 99999   # Roi (arbitrairement très élevé)
}

def eval_board(board: np.ndarray) -> int:
    score = 0

    for pt in [1, 3, 4, 5, 7, 9]:
        mask_white = (board == pt)
        mask_black = (board == -pt)

        # Nombre de pièces trouvées
        count_white = np.count_nonzero(mask_white)
        count_black = np.count_nonzero(mask_black)

        material_white = piece_values[pt] * count_white
        material_black = piece_values[pt] * count_black


        positional_white = np.sum(PieceTable[pt] * mask_white)
        positional_black = np.sum(PieceTable[-pt] * mask_black)

        score += (material_white + positional_white)
        score -= (material_black + positional_black)

    return score

board_print()
print(eval_board(board))