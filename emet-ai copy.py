import re 
import os
import copy
import time
import random
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import cProfile

profiler = cProfile.Profile()
profiler.enable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

coordinate = {
    "a":8,
    "b":7,
    "c":6,
    "d":5,
    "e":4,
    "f":3,
    "g":2,
    "h":1,
    8:"a",
    7:"b",
    6:"c",
    5:"d",
    4:"e",
    3:"f",
    2:"g",
    1:"h"
}

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

piece = {
    0: "vide",
    1: "pion_blanc",
    5: "tour_blanc",
    4: "fou_blanc",
    3: "cavalier_blanc",
    9: "reine_blanc",
    7: "roi_blanc",
    -1: "pion_noir",
    -5: "tour_noir",
    -4: "fou_noir",
    -3: "cavalier_noir",
    -9: "reine_noir",
    -7: "roi_noir"
}
 
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

all_rewards = {}

def check_repetition():
    board_tensors = [torch.tensor(board, dtype=torch.float32).flatten().to(device) for board in list_game_board_move]

    if len(board_tensors) == 0:
        return False

    stacked_boards = torch.stack(board_tensors)

    unique_boards, counts = torch.unique(stacked_boards, dim=0, return_counts=True)

    if torch.any(counts >= 3):
        return True
    return False

def valid_pion_move(move,promotion):
  if move['end_value'] == 0: 
      if list_game_move:
        if (move['y_start_coordinate'] == 5 if move['start_value'] > 0 else move['y_start_coordinate'] == 3):
            if board[move['y_start_coordinate']][move['x_end_coordinate']] == (-1 if move['start_value'] > 0 else 1):
                if move['x_end_coordinate'] == move['x_start_coordinate']+1:
                    if list_game_move[-1] == [[(move['y_start_coordinate']+2 if move['start_value'] > 0 else move['y_start_coordinate']-2),move['x_end_coordinate']],[move['y_start_coordinate'],move['x_end_coordinate']]]:
                        return('en passant')
                elif move['x_end_coordinate'] == move['x_start_coordinate']-1:
                    if list_game_move[-1] == [[(move['y_start_coordinate']+2 if move['start_value'] > 0 else move['y_start_coordinate']-2),move['x_end_coordinate']],[move['y_start_coordinate'],move['x_end_coordinate']]]:
                        return('en passant')
      if move['x_end_coordinate'] == move['x_start_coordinate']: 
          if move['name_piece_coor1'] == "pion_blanc":     
              if move['y_end_coordinate'] == move['y_start_coordinate'] + 1 or (move['y_end_coordinate'] == move['y_start_coordinate'] + 2 and move['y_start_coordinate'] == 1 and board[move['y_start_coordinate'] + 1][move['x_start_coordinate']] == 0):
                  if move['y_end_coordinate'] == 7:
                    if promotion:
                        print()
                        print("----promotion mode 👑----")
                        while True:
                            promotion_piece = input("enter a piece: the queen: 'Q', the rook: 'R', the bishop: 'B' and the knight 'N'> ") 
                            print()
                            if promotion_piece == 'Q':
                                move["start_value"] = 9
                                return('valid')
                                
                            if promotion_piece == 'R':
                                move["start_value"] = 5
                                return('valid')

                            if promotion_piece == 'B':
                                move["start_value"] = 4
                                return('valid')

                            if promotion_piece == 'N':
                                move["start_value"] = 3
                                return('valid')
                    else:
                        move["start_value"] = 9
                        return('valid')
                  return('valid')
              else:
                  return('illegal')

          if move['name_piece_coor1'] == "pion_noir":
            if move['y_start_coordinate'] == move['y_end_coordinate'] + 1 or (move['y_start_coordinate'] == move['y_end_coordinate'] + 2 and move['y_start_coordinate'] == 6 and board[move['y_start_coordinate'] - 1][move['x_start_coordinate']] == 0):
                if move['y_end_coordinate'] == 0:
                    if promotion:
                        print()
                        print("----promotion mode 👑----")
                        while True:
                            promotion_piece = input("enter a piece: the queen: 'Q', the rook: 'R', the bishop: 'B' and the knight 'N'> ") 
                            print()
                            if promotion_piece == 'Q':
                                move["start_value"] = -9
                                return('valid')
                                
                            if promotion_piece == 'R':
                                move["start_value"] = -5
                                return('valid')

                            if promotion_piece == 'B':
                                move["start_value"] = -4
                                return('valid')

                            if promotion_piece == 'N':
                                move["start_value"] = -3
                                return('valid')  
                    else:
                        move["start_value"] = -9
                        return('valid')
                return('valid')
            else:
                return('illegal')

          if move['name_piece_coor1'] != "pion_noir" and move['name_piece_coor1'] != "pion_blanc":
              return('illegal')
      else: 
          return('illegal')

  elif move['end_value'] != 0:
      if move['name_piece_coor1'] == "pion_blanc":
          if move['end_value'] > 0:
              return('illegal')
          if (move['x_end_coordinate'] == move['x_start_coordinate'] + 1 or move['x_end_coordinate'] == move['x_start_coordinate'] - 1) and move['y_end_coordinate'] == move['y_start_coordinate'] + 1:
                if move['y_end_coordinate'] == 7:
                    if promotion:
                        print()
                        print("----promotion mode 👑----")
                        while True:
                            promotion_piece = input("enter a piece: the queen: 'Q', the rook: 'R', the bishop: 'B' and the knight 'N'> ") 
                            print()
                            if promotion_piece == 'Q':
                                move["start_value"] = 9
                                return('valid')
                                
                            if promotion_piece == 'R':
                                move["start_value"] = 5
                                return('valid')

                            if promotion_piece == 'B':
                                move["start_value"] = 4
                                return('valid')

                            if promotion_piece == 'N':
                                move["start_value"] = 3
                                return('valid')
                    else:
                        move["start_value"] = 9
                        return('valid')
                return('valid')
          else:
              return('illegal')


      if move['name_piece_coor1'] == "pion_noir":
          if move['end_value'] < 0:
              return('illegal')
          if (move['x_end_coordinate'] == move['x_start_coordinate'] + 1 or move['x_end_coordinate'] == move['x_start_coordinate'] - 1) and move['y_end_coordinate'] == move['y_start_coordinate'] - 1:
            if move['y_end_coordinate'] == 0:
                if promotion:
                    print()
                    print("----promotion mode 👑----")
                    while True:
                        promotion_piece = input("enter a piece: the queen: 'Q', the rook: 'R', the bishop: 'B' and the knight 'N'> ") 
                        print()
                        if promotion_piece == 'Q':
                            move["start_value"] = -9
                            return('valid')
                            
                        if promotion_piece == 'R':
                            move["start_value"] = -5
                            return('valid')

                        if promotion_piece == 'B':
                            move["start_value"] = -4
                            return('valid')

                        if promotion_piece == 'N':
                            move["start_value"] = -3
                            return('valid')  
                else:
                    move["start_value"] = -9
                    return('valid')
            return('valid')
          else:
            return('illegal')
      return

def valid_bishop_move(move):
    if (move['name_piece_coor1'] in ('fou_noir,reine_noir') and move['end_value'] < 0) or(move['name_piece_coor1'] in ('fou_blanc','reine_blanc') and move['end_value'] > 0):
        return('illegal')
    
    if abs(move['x_start_coordinate'] - move['x_end_coordinate']) == abs(move['y_start_coordinate'] - move['y_end_coordinate']):
        if abs(move['x_start_coordinate'] - move['x_end_coordinate']) == 1:
            return 'valid'
        
        x = 1 if move['x_start_coordinate'] < move['x_end_coordinate'] else -1
        y = 1 if move['y_start_coordinate'] < move['y_end_coordinate'] else -1

        x_start_coordinate = move['x_start_coordinate'] + x
        y_start_coordinate = move['y_start_coordinate'] + y

        while x_start_coordinate != move['x_end_coordinate']:
            if board[y_start_coordinate][x_start_coordinate] != 0:
                    return('illegal')
            x_start_coordinate += x
            y_start_coordinate += y
                 
        return('valid')
    else:
        return('illegal')

def valid_rook_move(move, debug=None):
    global rook_m
    if move['end_value'] == 0 or (move['end_value'] < 0 and move['start_value'] > 0) or (move['end_value'] > 0 and move['start_value'] < 0):
        if move['y_start_coordinate'] == move['y_end_coordinate']:
            for i in range(min(move['x_start_coordinate'], move['x_end_coordinate']), max(move['x_start_coordinate'], move['x_end_coordinate'])):
                if board[move['y_start_coordinate']][i] != 0 and [i] != [move['x_start_coordinate']] and [i] != [move['x_end_coordinate']]:
                    return('illegal')
            if move['start_value'] == 5:
                if move['x_start_coordinate'] == 0 and move['y_start_coordinate'] == 0:
                    rook_m['rook_white_castling'] = False
                elif move['x_start_coordinate'] == 7 and move['y_start_coordinate'] == 0:
                    rook_m['rook_big_white_castling'] = False
            elif move['start_value'] == -5:
                if move['x_start_coordinate'] == 0 and move['y_start_coordinate'] == 7:
                    rook_m['rook_black_castling'] = False
                elif move['x_start_coordinate'] == 7 and move['y_start_coordinate'] == 7:
                    rook_m['rook_big_black_castling'] = False
            return('valid')
        elif move['x_start_coordinate'] == move['x_end_coordinate']:
            for i in range(min(move['y_start_coordinate'], move['y_end_coordinate']), max(move['y_start_coordinate'], move['y_end_coordinate'])):
                if board[i][move['x_start_coordinate']] != 0 and [i] != [move['y_start_coordinate']] and [i] != [move['y_end_coordinate']]:
                    return('illegal')
            if move['start_value'] == 5:
                if move['x_start_coordinate'] == 0 and move['y_start_coordinate'] == 0:
                    rook_m['rook_big_white_castling'] = False
                elif move['x_start_coordinate'] == 7 and move['y_start_coordinate'] == 0:
                    rook_m['rook_white_castling'] = False
            elif move['start_value'] == -5:
                if move['x_start_coordinate'] == 0 and move['y_start_coordinate'] == 7:
                    rook_m['rook_big_black_castling'] = False
                elif move['x_start_coordinate'] == 7 and move['y_start_coordinate'] == 7:
                    rook_m['rook_black_castling'] = False
            return('valid')
    return 'illegal'

def valid_knight_move(move):
    if move['end_value'] == 0 or ((move['end_value'] > 0 and move['start_value'] < 0) or (move['end_value'] < 0 and move['start_value'] > 0)): 
        if abs(move['x_start_coordinate'] - move['x_end_coordinate']) == 2 and abs(move['y_start_coordinate'] - move['y_end_coordinate']) == 1:
            return('valid')
        elif abs(move['x_start_coordinate'] - move['x_end_coordinate']) == 1 and abs(move['y_start_coordinate'] - move['y_end_coordinate']) == 2:
            return('valid')
    else:
        return('illegal')

def valid_king_move(move, castling_white=False, castling_black=False,big_castling_black=False,big_castling_white=False):
    if move['end_value'] == 0 and (castling_white if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else castling_black) and abs(move['x_start_coordinate'] - move['x_end_coordinate']) == 2 and abs(move['y_start_coordinate'] - move['y_end_coordinate']) == 0 and board[move['y_end_coordinate']][move['x_end_coordinate']-1] in (-5,5):
        if is_check(('white' if board[move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else 'black'),board) != 'check':
            if board[move['y_end_coordinate']][move['x_end_coordinate']+1] == 0:
                new_board = copy.deepcopy(board)
                new_board[move["y_start_coordinate"]][info_move["x_start_coordinate"]] = 0
                new_board[move["y_start_coordinate"]][info_move["x_start_coordinate"]-1] = board[move["y_start_coordinate"]][move["x_start_coordinate"]]
                if is_check(('white' if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else 'black'), new_board) != 'check':
                    if (rook_m['rook_white_castling'] == True if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else rook_m['rook_black_castling'] == True):
                        return 'casting'
    elif move['end_value'] == 0 and (big_castling_white if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else big_castling_black) and abs(move['x_start_coordinate'] - move['x_end_coordinate']) == 2 and abs(move['y_start_coordinate'] - move['y_end_coordinate']) == 0 and board[move['y_end_coordinate']][move['x_end_coordinate']+2] in (-5,5):
        if is_check(('white' if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else 'black'),board) != 'check':
            if board[move['y_end_coordinate']][move['x_end_coordinate']+1] == 0 and board[move['y_end_coordinate']][move['x_end_coordinate']-1] == 0:
                new_board = copy.deepcopy(board)
                new_board[move["y_start_coordinate"]][info_move["x_start_coordinate"]] = 0
                new_board[move["y_start_coordinate"]][info_move["x_start_coordinate"]+1] = board[move["y_start_coordinate"]][move["x_start_coordinate"]]
                if is_check(('white' if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else 'black'), new_board) != 'check':
                    if (rook_m['rook_big_white_castling'] == True if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else rook_m['rook_big_black_castling'] == True):
                        return 'big_casting'
    if move['end_value'] == 0 or ((move['end_value'] < 0 and move['start_value'] > 0) or (move['end_value'] > 0 and move['start_value'] < 0)): 
        if abs(move['x_start_coordinate'] - move['x_end_coordinate']) <= 1 and abs(move['y_start_coordinate'] - move['y_end_coordinate']) <= 1:
            return('valid')
        else:
            return('illegal')
    else:
        return('illegal')

def give_move_info(all_move,debug=None):
    try:
        all_move = all_move.split(" ")

        start_move = all_move[0]
        end_move = all_move[1]


        x_start_coordinate = start_move[:1]
        y_start_coordinate = int(start_move[1:]) - 1


        if x_start_coordinate in coordinate:
            x_start_coordinate = int(coordinate[x_start_coordinate]) - 1

        else:
            return('illegal')

        start_value = board[y_start_coordinate][x_start_coordinate]

        if start_value == 0:
            return('illegal')

        x_end_coordinate = end_move[:1]
        y_end_coordinate = int(end_move[1:]) - 1

        if x_end_coordinate in coordinate:
            x_end_coordinate = int(coordinate[x_end_coordinate]) - 1
        else:
            print("invalide coordinate invalide")
            return('illegal')

        end_value = board[y_end_coordinate][x_end_coordinate]

        name_piece_coor1 = piece[start_value]
        name_piece_coor2 = piece[end_value]

        if debug:
            print("----move----")
            print(f'{start_move} and {end_move}')
            print()
            print("----coordonnées----")
            print("coordinate valide matrice 1:", x_start_coordinate,"and",y_start_coordinate)
            print("coordinate valide matrice 2:", x_end_coordinate,"and",y_end_coordinate)
            print()
            print("----piece----")
            print("start_value:",start_value,"end_value:",end_value)
            print("start_value:",name_piece_coor1,"end_value:",name_piece_coor2)
            print()

        move = {
            "start_move":start_move,
            "end_move":end_move,
            "x_start_coordinate":x_start_coordinate,
            "y_start_coordinate":y_start_coordinate,
            "x_end_coordinate":x_end_coordinate,
            "y_end_coordinate":y_end_coordinate,
            "start_value":start_value,
            "end_value":end_value,
            "name_piece_coor1":name_piece_coor1,
            "name_piece_coor2":name_piece_coor2
        }

        return(move)
    except:
        return('illegal')

def valid_queen_move(move):
    if valid_rook_move(move) == 'valid' or valid_bishop_move(move) == 'valid':
        return('valid')
    else:
        return('illegal')

def board_print(style,color,board):
    if style:
        board_rendu = [list(reversed([piece_note_style[e] for e in r])) for r in board] if color == 'white' else [[piece_note_style[e] for e in r] for r in board]
    else:    
        board_rendu = [list(reversed([piece_note[e] for e in r])) for r in board]
    if color == 'white':
        board_rendu.reverse()
    for row in board_rendu:
        print(row)

def find_piece(board,piece):
    for x in range(len(board)):
        for y in range(len(board[x])):
            if board[x][y] == piece:
                return (y+1,x+1)

def is_check(color, board_actual):

    board = np.array(board_actual, dtype=int)

    if color == 'white':
        king_value    =  7   # Roi blanc
        enemy_pawn    = -1   # Pion noir
        enemy_knight  = -3   # Cavalier noir
        enemy_bishop  = -4   # Fou noir
        enemy_rook    = -5   # Tour noire
        enemy_queen   = -9   # Dame noire

    elif color == 'black':
        king_value    = -7   # Roi noir
        enemy_pawn    =  1   # Pion blanc
        enemy_knight  =  3   # Cavalier blanc
        enemy_bishop  =  4   # Fou blanc
        enemy_rook    =  5   # Tour blanche
        enemy_queen   =  9   # Dame blanche

    king_positions = np.argwhere(board == king_value)
    if len(king_positions) == 0:
        return 'error'
    king_y, king_x = king_positions[0]

    def check_straight_line_threats(y, x, rook_val, queen_val):
        row = board[y, :]
        col = board[:, x]

        right_segment = row[x+1:]
        mask_right = (right_segment != 0)
        if np.any(mask_right):
            idx_block = np.argmax(mask_right)  # premier True
            piece_encountered = right_segment[idx_block]
            if piece_encountered in (rook_val, queen_val):
                return True

        left_segment = row[:x][::-1] 
        mask_left = (left_segment != 0)
        if np.any(mask_left):
            idx_block = np.argmax(mask_left)
            piece_encountered = left_segment[idx_block]
            if piece_encountered in (rook_val, queen_val):
                return True

        down_segment = col[y+1:]
        mask_down = (down_segment != 0)
        if np.any(mask_down):
            idx_block = np.argmax(mask_down)
            piece_encountered = down_segment[idx_block]
            if piece_encountered in (rook_val, queen_val):
                return True

        up_segment = col[:y][::-1]
        mask_up = (up_segment != 0)
        if np.any(mask_up):
            idx_block = np.argmax(mask_up)
            piece_encountered = up_segment[idx_block]
            if piece_encountered in (rook_val, queen_val):
                return True
        return False


    def check_diagonal_threats(y, x, bishop_val, queen_val):
        """Renvoie True si un bishop_val (fou) ou queen_val (dame) est rencontré 
           en premier obstacle sur l'une des 4 diagonales partant du roi."""
        N = 8

        dist_down  = N - 1 - y
        dist_right = N - 1 - x
        steps = min(dist_down, dist_right)
        diag_bd = [board[y + i, x + i] for i in range(1, steps+1)]
        # Premier obstacle
        diag_bd_arr = np.array(diag_bd)
        mask_bd = diag_bd_arr != 0
        if np.any(mask_bd):
            idx_block = np.argmax(mask_bd)
            if diag_bd_arr[idx_block] in (bishop_val, queen_val):
                return True

        dist_down = N - 1 - y
        dist_left = x
        steps = min(dist_down, dist_left)
        diag_bg = [board[y + i, x - i] for i in range(1, steps+1)]
        diag_bg_arr = np.array(diag_bg)
        mask_bg = diag_bg_arr != 0
        if np.any(mask_bg):
            idx_block = np.argmax(mask_bg)
            if diag_bg_arr[idx_block] in (bishop_val, queen_val):
                return True

        dist_up    = y
        dist_right = N - 1 - x
        steps = min(dist_up, dist_right)
        diag_hd = [board[y - i, x + i] for i in range(1, steps+1)]
        diag_hd_arr = np.array(diag_hd)
        mask_hd = diag_hd_arr != 0
        if np.any(mask_hd):
            idx_block = np.argmax(mask_hd)
            if diag_hd_arr[idx_block] in (bishop_val, queen_val):
                return True

        dist_up  = y
        dist_left = x
        steps = min(dist_up, dist_left)
        diag_hg = [board[y - i, x - i] for i in range(1, steps+1)]
        diag_hg_arr = np.array(diag_hg)
        mask_hg = diag_hg_arr != 0
        if np.any(mask_hg):
            idx_block = np.argmax(mask_hg)
            if diag_hg_arr[idx_block] in (bishop_val, queen_val):
                return True

        return False

    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                    (1, 2), (1, -2), (-1, 2), (-1, -2)]
    def check_knight_threats(y, x, knight_val):
        for dy, dx in knight_moves:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < 8 and 0 <= xx < 8:
                if board[yy, xx] == knight_val:
                    return True
        return False

    def check_pawn_threats(y, x, pawn_val):

        if pawn_val == 1: 
            candidate_squares = [(y-1, x-1), (y-1, x+1)]
        else:          
            candidate_squares = [(y+1, x-1), (y+1, x+1)]

        for (yy, xx) in candidate_squares:
            if 0 <= yy < 8 and 0 <= xx < 8:
                if board[yy, xx] == pawn_val:
                    return True
        return False

    def check_king_threat(y, x):
        king_moves = [(1, 1), (-1, 1), (-1, -1), (1, -1),
                      (0, 1),  (0, -1),  (-1, 0),  (1, 0)]
        for (dy, dx) in king_moves:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < 8 and 0 <= xx < 8:
                if abs(board[yy, xx]) == 7:
                    # Le code initial vérifiait s’il y avait un roi ±7
                    return True
        return False

    if check_straight_line_threats(king_y, king_x, enemy_rook, enemy_queen):
        return 'check'

    if check_diagonal_threats(king_y, king_x, enemy_bishop, enemy_queen):
        return 'check'

    if check_knight_threats(king_y, king_x, enemy_knight):
        return 'check'

    if check_pawn_threats(king_y, king_x, enemy_pawn):
        return 'check'

    if check_king_threat(king_y, king_x):
        return 'check'

    return 'valid'

def list_pion_move(y,x):
    list_p_move = []
    direction = 1 if board[y][x] == 1 else -1 
    if board[y + direction][x] == 0:
        list_p_move.append([[y,x],[y + direction, x]])
        if (y == 1 and direction == 1) or (y == 6 and direction == -1):
            if board[y + 2 * direction][x] == 0:
                list_p_move.append([[y,x],[y + 2 * direction, x]])
    if x + 1 <= 7 and board[y + direction][x + 1] * board[y][x] < 0:
        list_p_move.append([[y,x],[y + direction, x + 1]])
    if x - 1 >= 0 and board[y + direction][x - 1] * board[y][x] < 0: 
        list_p_move.append([[y,x],[y + direction, x - 1]])
    
    return list_p_move

def list_knight_move(y,x):
    list_k_move = []
    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),(1, 2), (1, -2), (-1, 2), (-1, -2)]
    if board[y][x] in (3,-3):
        for e in knight_moves:
            if y+e[0] >= 0 and y+e[0] <= 7 and x+e[1] >= 0 and x+e[1] <= 7:
                if (board[y+e[0]][x+e[1]] <= 0 if board[y][x] == 3 else board[y+e[0]][x+e[1]] >= 0):
                    list_k_move.append([[y,x],[y+e[0],x+e[1]]])
    return list_k_move

def list_bishop_move(y,x):
    list_b_move = []
    bishop_move = [(1,1),(-1,1),(-1,-1),(1,-1)]
    if board[y][x] in (4,-4):
        for e in bishop_move:
            y_c = y + e[0]
            x_c = x + e[1]
            while y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and board[y_c][x_c] == 0:
                list_b_move.append([[y,x],[y_c,x_c]])
                y_c = y_c + e[0]
                x_c = x_c + e[1]
            if y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and (board[y_c][x_c] < 0 if board[y][x] == 4 else board[y_c][x_c] > 0):
                list_b_move.append([[y,x],[y_c,x_c]])
    return list_b_move

def list_rook_move(y,x):
    list_r_move = []
    rook_move = [(0,1),(0,-1),(-1,0),(1,0)]
    if board[y][x] in (5,-5):
        for e in rook_move:
            y_c = y + e[0]
            x_c = x + e[1]
            while y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and board[y_c][x_c] == 0:
                list_r_move.append([[y,x],[y_c,x_c]])
                y_c = y_c + e[0]
                x_c = x_c + e[1]
            if y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and (board[y_c][x_c] < 0 if board[y][x] == 5 else board[y_c][x_c] > 0):
                list_r_move.append([[y,x],[y_c,x_c]])       
    return list_r_move

def list_queen_move(y,x):
    list_q_move = []
    queen_move = [(1,1),(-1,1),(-1,-1),(1,-1),(0,1),(0,-1),(-1,0),(1,0)]
    if board[y][x] in (9,-9):
        for e in queen_move:
            y_c = y + e[0]
            x_c = x + e[1]
            while y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and board[y_c][x_c] == 0:
                list_q_move.append([[y,x],[y_c,x_c]])
                y_c = y_c + e[0]
                x_c = x_c + e[1]
            if y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and (board[y_c][x_c] < 0 if board[y][x] == 9 else board[y_c][x_c] > 0):
                list_q_move.append([[y,x],[y_c,x_c]])       
    return list_q_move

def list_king_move(y,x):
    list_k_move = []
    king_move = [(1,1),(-1,1),(-1,-1),(1,-1),(0,1),(0,-1),(-1,0),(1,0)]
    if board[y][x] in (7,-7):
        for e in king_move:
            y_c = y + e[0]
            x_c = x + e[1]
            if y_c >= 0 and y_c <= 7 and x_c >= 0 and x_c <= 7 and (board[y_c][x_c] >= 0 if board[y][x] == -7 else board[y_c][x_c] <= 0):
                list_k_move.append([[y,x],[y_c,x_c]])
    if (castling_p_white if board[y][x] > 0 else castling_p_black) and (rook_m['rook_white_castling'] if board[y][x] > 0 else rook_m['rook_black_castling']) and board[y][x] > 0:
        if board[y][x-1] == 0 and board[y][x-2] == 0:
            new_board = copy.deepcopy(board)
            new_board[y][x-1] = board[y][x]
            new_board[y][x] = 0
            if is_check('white' if board[y][x] > 0 else 'black', new_board) != 'check':
                list_k_move.append([[y,x],[y,x-2]])
    elif (big_castling_p_white if board[y][x] > 0 else big_castling_p_black) and (rook_m['rook_big_white_castling'] if board[y][x] > 0 else rook_m['rook_big_black_castling']) and board[y][x] > 0:
        if board[y][x+1] == 0 and board[y][x+2] == 0 and board[y][x+3] == 0:
            new_board = copy.deepcopy(board)
            new_board[y][x+1] = board[y][x] 
            new_board[y][x] = 0
            if is_check('white' if board[y][x] > 0 else 'black', new_board) != 'check':
                list_k_move.append([[y,x],[y,x+2]])
    return list_k_move

def list_all_legal_move():
    global board
    list_all_move = []
    for y_i in range(0,8):
        for x_i in range(0,8):
            if board[y_i][x_i] != 0:
                n_move = []
                if board[y_i][x_i] in (1,-1):
                    n_move = list_pion_move(y_i,x_i)
                elif board[y_i][x_i] in (7,-7):
                    n_move = list_king_move(y_i,x_i)
                elif board[y_i][x_i] in (9,-9):
                    n_move = list_queen_move(y_i,x_i)
                elif board[y_i][x_i] in (5,-5):
                    n_move = list_rook_move(y_i,x_i)
                elif board[y_i][x_i] in (4,-4):
                    n_move = list_bishop_move(y_i,x_i)
                elif board[y_i][x_i] in (3,-3):
                    n_move = list_knight_move(y_i,x_i)
                if n_move:
                    for m in n_move:
                        new_board = copy.deepcopy(board)
                        new_board[m[0][0]][m[0][1]] = 0
                        new_board[m[1][0]][m[1][1]] = board[y_i][x_i]
                        if is_check('white' if board[y_i][x_i] > 0 else 'black', new_board) != 'check':
                            list_all_move.append(m)
    return list_all_move

def list_all_legal_white_move():
    global board
    list_all_move = []
    for y_i in range(0,8):
        for x_i in range(0,8):
            if board[y_i][x_i] != 0:
                n_move = []
                if board[y_i][x_i] == 1:
                    n_move = list_pion_move(y_i,x_i)
                elif board[y_i][x_i] == 7:
                    n_move = list_king_move(y_i,x_i)
                elif board[y_i][x_i] == 9:
                    n_move = list_queen_move(y_i,x_i)
                elif board[y_i][x_i] == 5:
                    n_move = list_rook_move(y_i,x_i)
                elif board[y_i][x_i] == 4:
                    n_move = list_bishop_move(y_i,x_i)
                elif board[y_i][x_i] == 3:
                    n_move = list_knight_move(y_i,x_i)
                if n_move:
                    for m in n_move:
                        new_board = copy.deepcopy(board)
                        new_board[m[0][0]][m[0][1]] = 0
                        new_board[m[1][0]][m[1][1]] = board[y_i][x_i]
                        if is_check('white', new_board) != 'check':
                            list_all_move.append(m)
    return list_all_move

def list_all_legal_black_move():
    global board
    list_all_move = []
    for y_i in range(0,8):
        for x_i in range(0,8):
            if board[y_i][x_i] != 0:
                n_move = []
                if board[y_i][x_i] == -1:
                    n_move = list_pion_move(y_i,x_i)
                elif board[y_i][x_i] == -7:
                    n_move = list_king_move(y_i,x_i)
                elif board[y_i][x_i] == -9:
                    n_move = list_queen_move(y_i,x_i)
                elif board[y_i][x_i] == -5:
                    n_move = list_rook_move(y_i,x_i)
                elif board[y_i][x_i] == -4:
                    n_move = list_bishop_move(y_i,x_i)
                elif board[y_i][x_i] == -3:
                    n_move = list_knight_move(y_i,x_i)
                if n_move:
                    for m in n_move:
                        new_board = copy.deepcopy(board)
                        new_board[m[0][0]][m[0][1]] = 0
                        new_board[m[1][0]][m[1][1]] = board[y_i][x_i]

                        if is_check('black', new_board) != 'check':
                            list_all_move.append(m)
    return list_all_move

def is_checkmate(color, board_actual):
    if color in ('black','white'):
        if is_check(color, board_actual) == 'check':
            move = list_all_legal_black_move() if color == 'black' else list_all_legal_white_move()
            for m in move:
                if board[m[0][0]][m[0][1]] not in (7,-7) or abs(m[0][1]-m[1][1]) != 2:
                    new_board = copy.deepcopy(board_actual)
                    new_board[m[0][0]][m[0][1]] = 0
                    new_board[m[1][0]][m[1][1]] = board_actual[m[0][0]][m[0][1]]
                    if is_check(color, new_board) != 'check':
                        return False
            return True
    return False

def play_move(print_board=False):
    global board
    global castling_p_white,castling_p_black,big_castling_p_white,big_castling_p_black,list_game_move
    result_valid_king = None
    result_valid_pion = None

    # if is_checkmate(color='black',board_actual=board):
    #     print("✅---checkmate white win---✅")
    #     return 'checkmate'
    # if is_checkmate(color='white',board_actual=board):
    #     print("✅---checkmate black win---✅")
    #     return 'checkmate'
    if info_move["start_value"] > 0:
        if is_check('black',board) != 'valid':
            print("🚫---invalide move black is in check---🚫")
            return
    else:
        if is_check('white',board) != 'valid':
            print("🚫---invalide move white is in check---🚫")
            return

    if info_move["start_value"] == 1 or info_move["start_value"] == -1:
        result_valid_pion = valid_pion_move(info_move,False)
        if result_valid_pion != 'valid' and result_valid_pion != 'en passant':
            print("🚫---invalide move---🚫")
            return
        elif result_valid_pion == 'en passant':
            new_board = copy.deepcopy(board)
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] = 0
            new_board[info_move["y_end_coordinate"]][info_move["x_end_coordinate"]] = info_move["start_value"]
            new_board[info_move["y_start_coordinate"]][info_move["x_end_coordinate"]] = 0
    elif info_move["start_value"] == 5 or info_move["start_value"] == -5:
        if valid_rook_move(info_move,debug=None) != 'valid':
            print("🚫---invalide move---🚫")
            return 
    elif info_move["start_value"] == 4 or info_move["start_value"] == -4:
        if valid_bishop_move(info_move) != 'valid':
            print("🚫---invalide move---🚫")
            return
    elif info_move["start_value"] == 3 or info_move["start_value"] == -3:
        if valid_knight_move(info_move) != 'valid':
            print("🚫---invalide move---🚫")
            return
    elif info_move["start_value"] == 7 or info_move["start_value"] == -7:
        result_valid_king = valid_king_move(info_move,castling_white=castling_p_white,castling_black=castling_p_black,big_castling_black=big_castling_p_black,big_castling_white=big_castling_p_white)
        if result_valid_king not in ['casting', 'big_casting','valid']:
            print("🚫---invalide move---🚫")
            return 
        elif result_valid_king == 'casting':
            new_board = copy.deepcopy(board)
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] = 0
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]-3] = 0
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]-1] = (5 if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else -5)
            new_board[info_move["y_end_coordinate"]][info_move["x_end_coordinate"]] = info_move["start_value"]
        elif result_valid_king == 'big_casting':
            new_board = copy.deepcopy(board)
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] = 0
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]+4] = 0
            new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]+1] = (5 if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0 else -5)
            new_board[info_move["y_end_coordinate"]][info_move["x_end_coordinate"]] = info_move["start_value"]
        else:
            if info_move["start_value"] > 0:
                castling_p_white, big_castling_p_white = (False, False)  
            else:
                castling_p_black, big_castling_p_black = (False, False)   

    elif info_move["start_value"] == 9 or info_move["start_value"] == -9:
        if valid_queen_move(info_move) != 'valid':
            print("🚫---invalide move---🚫")
            return
    
    if result_valid_king not in ['big_casting', 'casting'] and result_valid_pion != 'en passant':
        new_board = copy.deepcopy(board)
        new_board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] = 0
        new_board[info_move["y_end_coordinate"]][info_move["x_end_coordinate"]] = info_move["start_value"]
    elif result_valid_king  != 'big_casting':
        if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0: 
            castling_p_white = False
        else:
            castling_p_black = False
    else:
        if board[info_move["y_start_coordinate"]][info_move["x_start_coordinate"]] > 0: 
            big_castling_p_white = False
        else:
            big_castling_p_black = False


    if info_move["start_value"] < 0:
        if is_check('black',new_board) != 'valid':
            print("🚫---invalide move black is in check---🚫")
            return
    else:
        if is_check('white',new_board) != 'valid':
            print("🚫---invalide move white is in check---🚫")
            return
    
    board = copy.deepcopy(new_board)
    
    if is_checkmate(color='black',board_actual=board):
        if print_board:
            print()
            print("✅---checkmate white win---✅")
        if print_board:
            print()
        list_game_board_move.append(board)
        list_game_move.append([[info_move['y_start_coordinate'],info_move["x_start_coordinate"]],[info_move['y_end_coordinate'], info_move['x_end_coordinate']]])
        if print_board:
            board_print(True,'black',board)
        return 'checkmate'
    if is_checkmate(color='white',board_actual=board):
        if print_board:
            print()
            print("✅---checkmate black win---✅")
        if print_board:
            print()
        list_game_board_move.append(board)
        list_game_move.append([[info_move['y_start_coordinate'], info_move['x_start_coordinate']],[info_move['y_end_coordinate'], info_move['x_end_coordinate']]])
        if print_board:   
            board_print(True,'white',board)
        return 'checkmate'
    if print_board:
        print("✅---valide move---✅")
        print()
    list_game_move.append([[info_move['y_start_coordinate'], info_move['x_start_coordinate']],[info_move['y_end_coordinate'], info_move['x_end_coordinate']]])
    list_game_board_move.append(board)
    return 'valid'

def play():
    global castling_p_white,castling_p_black,big_castling_p_white,big_castling_p_black,info_move,rook_m,list_game_move,list_game_board_move
    global board
    castling_p_white = True
    castling_p_black = True 
    big_castling_p_white = True 
    big_castling_p_black = True
    rook_m = {'rook_white_castling' : True, 'rook_black_castling' : True, 'rook_big_white_castling' : True, 'rook_big_black_castling' : True}
    list_game_move =[]
    list_game_board_move =[]

    board = [
        [ 5, 3, 4, 7, 9, 4, 3, 5],
        [ 1, 1, 1, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-5,-3,-4,-7,-9,-4,-3,-5]
    ]

    if board[0][3] != 7:
        castling_p_white = False
        big_castling_p_white = False
    if board[0][0] != 5:
        rook_m['rook_white_castling'] = False
    if board[0][7] != 5:
        rook_m['rook_big_white_castling'] = False
    if board[7][3] != -7:
        castling_p_black = False
        big_castling_p_black = False
    if board[7][0] != -5:
        rook_m['rook_black_castling'] = False
    if board[7][7] != -5:
        rook_m['rook_big_black_castling'] = False

    print(list_all_legal_white_move())
    print(len(list_all_legal_white_move()))
    print("🌟 Game start 🌟")
    print()
    print("⚪---white play---⚪")
    print()
    board_print(True,'white',board)
    while True:
        for i in range(0,2):
            rep = None
            while rep != 'valid':
                # time.sleep(2)
                # while True:
                #     print()
                #     all_move = input(">")
                #     pvrint()
                #     if bool(re.match(r'^[a-h][1-8]\s[a-h][1-8]$', all_move)):
                #         break
                #     else:
                #         print("🚫---invalide move---🚫 => valide move example: ✅--- e2 e4 ---✅")
                

                # i = 1
                # move = [[0, 0], [0, 2]]
                # all_move = "h1 f1"   
                
                legal_white_move = list_all_legal_white_move()
                legal_black_move = list_all_legal_black_move()


                if legal_white_move == [] and i == 0:
                    print("⬛---whites are pat---⬛")
                    print()
                    board_print(True,'white',board)
                    return 'pat'
                if legal_black_move == [] and i == 1:
                    print("⬛---blacks are pat---⬛")
                    print()
                    board_print(True,'black',board)
                    print()
                    return 'pat'


                # if i == 1:
                #     move = random.choice(list_all_legal_white_move()) if i == 0 else random.choice(list_all_legal_black_move())
                #     all_move = f"{coordinate[move[0][1]+1]}{move[0][0]+1} {coordinate[move[1][1]+1]}{move[1][0]+1}"
                # else:
                #     move = []
                #     all_move = input(">")    

                move = random.choice(list_all_legal_white_move()) if i == 0 else random.choice(list_all_legal_black_move())
                all_move = f"{coordinate[move[0][1]+1]}{move[0][0]+1} {coordinate[move[1][1]+1]}{move[1][0]+1}"   
                
                print(all_move,move)
                
                info_move = give_move_info(all_move,debug=None)
                if info_move == 'illegal':
                    print("🚫---invalide move---🚫")
                elif (info_move['start_value'] > 0 and i == 1) or (info_move['start_value'] < 0 and i == 0):
                    print("🚫---It's not your turn ---🚫")
                else:
                    rep = play_move()

                if rep == 'checkmate':
                    return 'checkmate'

            if check_repetition():
                if i == 0:
                    board_print(True,'white',board)
                else:
                    board_print(True,'black',board)
                if print_board:
                    print("⏸---Draw by repetition---⏸")
                return 'draw'
            
            if len(list_game_move) >= 50:
                no_capture_moves = all(board[move[1][0]][move[1][1]] == 0 for move in list_game_move[-50:])               

                if no_capture_moves:
                    if i == 0:
                        board_print(True,'white',board)
                    else:
                        board_print(True,'black',board)
                    if print_board:
                        print("⏸---Draw by fifty-move rule---⏸")
                    return 'draw'

            def material_insufficiency(board):
                for row in board:
                    for element in row:
                        if element not in [0, 7, -7]:
                            return False
                return True

            if material_insufficiency(board):
                if print_board:
                    print("⏸---Draw by insufficient material---⏸")
                    print()
                if i == 0:
                    board_print(True,'white',board) 
                else:
                    board_print(True,'black',board)
                return 'draw'
                
            if i == 0:
                print()
                print("⚫---black play---⚫")
                print()
                board_print(True,'black',board)

            elif i == 1:
                print()
                print("⚪---white play---⚪")
                print()
                board_print(True,'white',board)    

# Hyperparamètres
epsilon = 0.999
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99
lr = 0.0005
batch_size = 2048
replay_buffer_size = 1000000
update_target_every = 500
episode = 10
model_name = "chess_ai_v3_100000_epi"

piece_reward = {
    1: 1,   
    3: 3,  
    4: 3,  
    5: 5,   
    9: 9
}

stats = {
    "wins_white": 0,
    "wins_black": 0,
    "draws": 0,
    "average_game_length": [],
    "epsilon_values": []
}

reward_check = 5
nobonus = -0.1

class chess_ia(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(chess_ia, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.3)  
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)  

        self.fc3 = nn.Linear(256, output_dim)  

        self._initialize_weights()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  

        # Fully Connected avec Dropout
        x = F.leaky_relu(self.dropout1(self.fc1(x)))
        x = F.leaky_relu(self.dropout2(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

def eval_board(board: np.ndarray) -> int:
    score = 0

    for pt in [1, 3, 4, 5, 7, 9]:
        mask_white = (board == pt)
        mask_black = (board == -pt)
        count_white = np.count_nonzero(mask_white)
        count_black = np.count_nonzero(mask_black)
        material_white = piece_values[pt] * count_white
        material_black = piece_values[pt] * count_black
        positional_white = np.sum(PieceTable[pt] * mask_white)
        positional_black = np.sum(PieceTable[-pt] * mask_black)

        score += (material_white + positional_white)
        score -= (material_black + positional_black)

    return score
def choose_move(state):
    if random.random() < epsilon:
        return random.choice(legal_moves)
    else:
        # Conversion explicite du tenseur vers CUDA
        if torch.is_tensor(state):
            state_tensor = state.clone().detach().to(dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state_tensor = torch.from_numpy(np.array(state)).to(dtype=torch.float32, device=device).unsqueeze(0)

        model.to(device)

        q_values = model(state_tensor).squeeze()

        legal_q_values = torch.tensor([], device=device)

        for move_index, move in enumerate(legal_moves):
            if move_index < len(q_values):
                legal_q_values = torch.cat([legal_q_values, q_values[move_index].unsqueeze(0)])

        if legal_q_values.numel() == 0:
            return random.choice(legal_moves)
        
        best_move_index = legal_q_values.argmax()
        best_move = legal_moves[best_move_index]
        
        return best_move

input_dim = 3
output_dim = 467
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
model = chess_ia(input_dim, output_dim).to(device)

if hasattr(torch, 'compile'):
    try:
        torch.set_float32_matmul_precision('high')

        model = torch.compile(model)
        print("✅ Modèle compilé avec succès grâce à torch.compile.")
    except Exception as e:
        print(f"⚠️ Impossible de compiler le modèle avec torch.compile. Erreur : {e}")
else:
    print("❌ torch.compile non disponible dans cette version de PyTorch.")
        
optimizer = optim.Adam(model.parameters(), lr=lr)
replay_buffer = ReplayBuffer(replay_buffer_size)
criterion = nn.MSELoss()

list_r_episode = []
for epi in range(episode):
    all_rewards[f"episode_{epi+1}"] = []
    castling_p_white = True
    castling_p_black = True 
    big_castling_p_white = True 
    big_castling_p_black = True
    rook_m = {'rook_white_castling' : True, 'rook_black_castling' : True, 'rook_big_white_castling' : True, 'rook_big_black_castling' : True}
    list_game_move =[]
    list_game_board_move = []
    black_reward = 0
    white_reward = 0
    global_black_reward = 0
    global_white_reward = 0
    print_board = False
    board = [
        [ 5, 3, 4, 7, 9, 4, 3, 5],
        [ 1, 1, 1, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-5,-3,-4,-7,-9,-4,-3,-5]
    ]
    
    if board[0][3] != 7:
        castling_p_white = False
        big_castling_p_white = False
    if board[0][0] != 5:
        rook_m['rook_white_castling'] = False
    if board[0][7] != 5:
        rook_m['rook_big_white_castling'] = False
    if board[7][3] != -7:
        castling_p_black = False
        big_castling_p_black = False
    if board[7][0] != -5:
        rook_m['rook_black_castling'] = False
    if board[7][7] != -5:
        rook_m['rook_big_black_castling'] = False
    if print_board:
        print(list_all_legal_white_move())
        print(len(list_all_legal_white_move()))
        print("🌟 Game start 🌟")
        print()
        print("⚪---white play---⚪")
        print()
        board_print(True,'white',board)
    rep_f = []
    while rep_f not in ['checkmate','draw','pat']:
        for i in range(0,2):
            rep = None
            legal_white_move = list_all_legal_white_move()
            legal_black_move = list_all_legal_black_move()
            
            # Création des tenseurs directement sur GPU
            board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
            legal_moves = legal_white_move if i == 0 else legal_black_move
            
            mask = torch.zeros((8, 8), dtype=torch.float32, device=device)
            for move in legal_moves:
                start, end = move
                mask[end[0], end[1]] = 1
            legal_tensor = mask.unsqueeze(0)
            
            mask_game_move = torch.zeros((8, 8), dtype=torch.float32, device=device)
            if list_game_move:
                last_move = list_game_move[-1]
                if len(last_move) == 2:
                    start, end = last_move
                    mask_game_move[start[0], start[1]] = 1
                    mask_game_move[end[0], end[1]] = 1
            
            tensor_game_move = mask_game_move.unsqueeze(0)
            # Concaténation des tenseurs (déjà sur GPU)
            state = torch.cat([board_tensor, legal_tensor, tensor_game_move], dim=0)
            while rep != 'valid':

                if legal_white_move == [] and i == 0:
                    if print_board:
                        print()
                        print("⬛---whites are pat---⬛")
                    if print_board:
                        print()
                    if print_board:
                        board_print(True,'white',board)
                    list_game_board_move.append(board)
                    list_game_move.append([[info_move['y_start_coordinate'], info_move['x_start_coordinate']],[info_move['y_end_coordinate'], info_move['x_end_coordinate']]])
                    rep_f = 'pat'
                    break
                if legal_black_move == [] and i == 1:
                    if print_board:
                        print()
                        print("⬛---blacks are pat---⬛")
                    if print_board:
                        print()
                    if print_board:
                        board_print(True,'black',board)
                    list_game_move.append([[info_move['y_start_coordinate'], info_move['x_start_coordinate']],[info_move['y_end_coordinate'], info_move['x_end_coordinate']]])
                    list_game_board_move.append(board)
                    rep_f = 'pat'
                    reward = -10
                    done = True
                    all_rewards[f"episode_{epi+1}"].append(reward)
                    replay_buffer.push(state, action, reward, next_state, done)
                    break
                
                ex_board = board
                move = choose_move(state)
                all_move = f"{coordinate[move[0][1]+1]}{move[0][0]+1} {coordinate[move[1][1]+1]}{move[1][0]+1}"   
                if print_board:
                    print(all_move,move)
                
                info_move = give_move_info(all_move,debug=None)
                if info_move == 'illegal':
                    print("🚫---invalide move---🚫")
                elif (info_move['start_value'] > 0 and i == 1) or (info_move['start_value'] < 0 and i == 0):
                    print("🚫---It's not your turn ---🚫")
                else:
                    rep = play_move()
            
                board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)
                legal_moves = list_all_legal_white_move() if i == 0 else list_all_legal_black_move()
                mask = torch.zeros((8, 8), dtype=torch.float32)
                for move in legal_moves:
                    start, end = move 
                    mask[end[0], end[1]] = 1 
                legal_tensor = mask.unsqueeze(0)
                mask_game_move = torch.zeros((8, 8), dtype=torch.float32)
                if list_game_move:
                    last_move = list_game_move[-1]
                    if len(last_move) == 2:
                        start, end = last_move
                        mask_game_move[start[0], start[1]] = 1
                        mask_game_move[end[0], end[1]] = 1

                tensor_game_move = mask_game_move.unsqueeze(0)
                board_tensor = board_tensor.to(device)
                legal_tensor = legal_tensor.to(device)
                tensor_game_move = tensor_game_move.to(device)
                next_state = torch.cat([board_tensor,legal_tensor,tensor_game_move], dim=0).to(device)
                action = move



                if rep == 'checkmate':
                    rep_f ='checkmate' 
                    reward = 100
                    done = True
                    all_rewards[f"episode_{epi+1}"].append(reward)
                    replay_buffer.push(state, action, reward, next_state, done)
                    break
                if check_repetition():
                    if i == 0:
                        if print_board:
                            board_print(True,'white',board)
                    else:
                        if print_board:
                            board_print(True,'black',board)
                    if print_board:
                        print()
                        print("⏸---Draw by repetition---⏸")
                    if print_board:
                        print()
                    rep_f = 'draw'
                    reward = -10
                    done = True
                    all_rewards[f"episode_{epi+1}"].append(reward)
                    replay_buffer.push(state, action, reward, next_state, done)
                    break

                if len(list_game_move) >= 50:
                    no_capture_moves = all(board[move[1][0]][move[1][1]] == 0 for move in list_game_move[-50:])               

                    if no_capture_moves:
                        if i == 0:
                            if print_board:
                                board_print(True,'white',board)
                        else:
                            if print_board:
                                board_print(True,'black',board)
                        print("⏸---Draw by fifty-move rule---⏸")
                        rep_f = 'draw'

                def material_insufficiency(board):
                    for row in board:
                        for element in row:
                            if element not in [0, 7, -7]:
                                return False
                    return True

                if material_insufficiency(board):
                    if print_board:
                        print()
                        print("⏸---Draw by insufficient material---⏸")
                    if print_board:
                        print()
                    if i == 0:
                        if print_board:
                            board_print(True,'white',board)
                    else:
                        if print_board:
                            board_print(True,'black',board)
                    rep_f = 'draw'
                
                if rep_f == 'draw' or rep == 'pat':
                    reward = -10
                    done = True
                    all_rewards[f"episode_{epi+1}"].append(reward)
                    replay_buffer.push(state, action, reward, next_state, done)
                    break

                reward = 0 
                if i == 0:
                    if is_check('black',board) == 'check':
                        reward += reward_check
                else:
                    if is_check('white',board) == 'check':
                        reward += reward_check
                
                if reward == 0:
                    reward = nobonus

                # board_tensor_current = state[0]  
                # board_tensor_next = next_state[0]
                # board_diff = board_tensor_next - board_tensor_current 
                # for y in range(8):
                #     for x in range(8):
                #         diff = board_diff[y, x].item()  
                #         reward += abs(piece_reward.get(board_tensor_current[y, x].item(), 0))
                   
                reward += (eval_board(ex_board) - eval_board(board)) / 100 #exprimé en centi-pions
                if print_board:
                    print("ex reward",reward)
                    print(f"{eval_board(ex_board)} - {eval_board(board)}")
                    print("now reward",reward)
                done = False
                all_rewards[f"episode_{epi+1}"].append(reward)
                replay_buffer.push(state, action, reward, next_state, done)

            if rep_f in ['checkmate','draw','pat']:
                break
                
            if i == 0:
                if print_board:
                    print()
                    print("⚫---black play---⚫")
                    print()
                    board_print(True,'black',board)

            elif i == 1:
                if print_board:
                    print()
                    print("⚪---white play---⚪")
                    print()
                    board_print(True,'white',board)  
        
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    list_r_episode.append(rep_f)
    update_epi = 10
    if (epi + 1) % update_epi == 0:
        print(f"Episode {epi+1}, Epsilon: {epsilon},reward: {reward}")
    if (epi + 1) % 100 == 0:
        save_path = f"{model_name}-save-{epi + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Modèle sauvegardé à l'épisode {epi + 1} sous {save_path}")
    
    stats["epsilon_values"].append(epsilon)
    if rep_f == "checkmate":
        if i == 0:
            stats["wins_white"] += 1
        else:
            stats["wins_black"] += 1
    elif rep_f == "draw":
        stats["draws"] += 1
    stats["average_game_length"].append(len(list_game_move))

torch.save(model.state_dict(), f"{model_name}.pth")
print("Modèle save")
print(list_r_episode)

os.mkdir(model_name)

# plt.plot(stats["epsilon_values"])
# plt.xlabel("Épisodes")
# plt.ylabel("Epsilon")
# plt.title("Évolution d'Epsilon")
# plt.savefig(f"{model_name}/stats_epsilon_values.png")
# plt.close()

# labels = ["Wins White", "Wins Black", "Draws"]
# values = [stats["wins_white"], stats["wins_black"], stats["draws"]]

# plt.bar(labels, values)
# plt.title("Distribution des résultats")
# plt.savefig(f"{model_name}/stats_results_distribution.png")
# plt.close()

# plt.plot(stats["average_game_length"])
# plt.xlabel("Épisodes")
# plt.ylabel("Longueur moyenne des parties")
# plt.title("Évolution de la longueur moyenne des parties")
# plt.savefig(f"{model_name}/stats_average_game_length.png")
# plt.close()

# plt.figure(figsize=(10, 6))
# for episode, rewards in all_rewards.items():
#     plt.plot(rewards, label=episode)

# plt.xlabel("Étapes dans la partie")
# plt.ylabel("Récompenses")
# plt.title("Évolution des récompenses au fil des parties")
# plt.legend()
# plt.savefig(f"{model_name}/rewards_comparison.png")
# plt.close()

with open(f"{model_name}/all_rewards.json", "w") as json_file:
    json.dump(all_rewards, json_file, indent=4)
with open(f"{model_name}/stats.json", "w") as json_file:    
    json.dump(stats, json_file, indent=4)

profiler.disable()
profiler.dump_stats("profiling_results2.prof")
print("Profilage terminé, fichier créé.")