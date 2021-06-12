import chess
import chess.variant
import chess.pgn
import numpy as np

#pgn = open("sampleg.pgn")

def encode_move(move):
    move_en=np.zeros((64,64))
    move_en[move.from_square,move.to_square]=1
    return move_en

def encode_promotion(move):
    prom=np.zeros(4)
    if move.promotion:
        prom[move.promotion-2]=1


def decode_move(move,game):
    move.reshape((64,64))
    move_dec=chess.SQUARE_NAMES[np.where(move==1)[0][0]] + chess.SQUARE_NAMES[np.where(move==1)[1][0]]
    for m in game.legal_moves:
        if move_dec in m.uci() and len(move_dec)!=len(m.uci()):
            move_dec=move_dec+'q'
            break
    return move_dec







