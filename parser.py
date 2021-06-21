import chess
import chess.variant
import chess.pgn
import numpy as np
import pickle
import sys
import joblib 
def move_parser(filename,filesave):
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

def result_parser(filename,filesave):
    pgn=open(filename)
    table = {}
    game=chess.pgn.read_game(pgn)
    board=chess.variant.AtomicBoard()
    counter=0
    while game:
        if game.headers["Result"]=="1-0": result=1
        elif game.headers["Result"]=="0-1": result=-1
        else: result=0
        for move in game.mainline_moves():
            if board.fen() in table:
                table[board.fen()]["result"]+=result
                table[board.fen()]["total"]+=1
            else:
                table[board.fen()]={}
                table[board.fen()]["result"]=result
                table[board.fen()]["total"]=1
            board.push(move)
        game=chess.pgn.read_game(pgn)
        board=chess.variant.AtomicBoard()
        print("finished game ",counter)
        counter+=1
    output = open(filesave, 'wb')
    joblib.dump(table, output)
    output.close()
    return 

if __name__ == "__main__":
    #move_parser("filtered_2200.pgn","move_table2200")
    result_parser("filtered_2200.pgn","result_table2200")
    #f = open("move_table2200","rb")
    #f = open("result_table2200","rb")
    #table= joblib.load(f)
    #print(table)
    #print(table[chess.variant.AtomicBoard().fen()])
    #print(len(table))
