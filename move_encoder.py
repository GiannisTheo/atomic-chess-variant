import chess
import chess.variant
import chess.pgn
import numpy as np



def encode_move(move):
    if len(move)!=4: move=move[:4]
    move_en=np.zeros((64,64))
    start=chess.SQUARE_NAMES.index(move[:2])
    end= chess.SQUARE_NAMES.index(move[2:])
    #move_en[move.from_square,move.to_square]=1
    move_en[start,end]=1
    return move_en.reshape(64*64)

def encode_promotion(move):
    prom=np.zeros(4)
    if move.promotion:
        prom[move.promotion-2]=1


def decode_move(move_en,game):
    move_en=move_en.reshape((64,64))
    move_dec=chess.SQUARE_NAMES[np.where(move_en==1)[0][0]] + chess.SQUARE_NAMES[np.where(move_en==1)[1][0]]
    for m in game.legal_moves:
        if move_dec in m.uci() and len(move_dec)!=len(m.uci()):
            move_dec=move_dec+'q'
            break
    return move_dec

def get_best_move(policy,game):
    move=np.zeros(64*64)
    move[policy.argmax()]=1
    return decode_move(move,game)



def mask_invalid(policy,position):
    legal_moves=np.zeros((64,64))
    for m in position.legal_moves:
        legal_moves[m.from_square,m.to_square]=1
    masked=policy.reshape((64,64))*legal_moves
    masked = masked/np.sum(masked)
    return masked.reshape(64*64)

if __name__ == '__main__':
    board=chess.variant.AtomicBoard()
    policy=np.random.random(64*64)
    p=mask_invalid(policy,board)
    print(get_best_move(p,board))
    pgn=open("sampleg.pgn")
    game=chess.pgn.read_game(pgn)
    board=chess.variant.AtomicBoard()
    while game:
        for move in game.mainline_moves():
            move_encoded=encode_move(move.uci())
            move_decoded= decode_move(move_encoded,board)
            print(move_decoded,move.uci())
            if move_decoded!=move.uci(): print("gamithike")
            board.push(move)
        game=chess.pgn.read_game(pgn)
        board=chess.variant.AtomicBoard()







