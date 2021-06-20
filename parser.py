import chess
import chess.variant
import chess.pgn
import numpy as np
import pickle
import sys
import joblib 
def parser(filename,filesave):
    pgn = open(filename)
    table = {}
    game=chess.pgn.read_game(pgn)
    board=chess.variant.AtomicBoard()
    counter=0
    while game:
        for mov in game.mainline_moves():
            move = mov.uci()[:4] #discard the promotion
            if not board.fen() in table:
                table[board.fen()]= {}
            if move in table[board.fen()]:
                table[board.fen()][move]+=1
            else: 
                table[board.fen()][move]=1
            board.push(mov)
        game=chess.pgn.read_game(pgn)
        board=chess.variant.AtomicBoard()
        print("finished game ",counter)
        counter+=1
    output = open(filesave, 'wb')
    joblib.dump(table, output)
    output.close()
    return 
#print(table)

if __name__ == "__main__":
    parser("filtered_2200.pgn","move_table2200")
    #f = open("testjoblib","rb")
    #table= joblib.load(f)
    #print(table)
    #print(table[chess.variant.AtomicBoard().fen()]["e2e4"])
    #print(len(table))
