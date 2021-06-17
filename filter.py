import chess
import chess.variant
import chess.pgn
import numpy as np 
import os
import sys

#pgn=open("last_month.pgn")
#os.chdir("./lichess_db")

def filter_games(elo):
    bullet_format= ["0+1","1+1","1/2+0","1+0"]
    for f in os.listdir():
        pgn=open(f)
        
        offsets = []
        while True:
            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break
            if "?" not in headers["WhiteElo"]:
                welo= int(headers["WhiteElo"])
            else: continue
            if  "?" not in headers["BlackElo"]:
                belo= int(headers["BlackElo"])
            else: continue

            if welo>=elo and belo>=elo:
                if headers["TimeControl"] in bullet_format: continue
                offsets.append(offset)
                #counter+=1
                #print(counter,i)
        
        for offset in offsets:
            pgn.seek(offset)
            game=chess.pgn.read_game(pgn)
            print(game)
            print()
            print()

if __name__ == "__main__":
    #os.chdir("./lichess_db")
    orig_stdout = sys.stdout
    output=open("filtered_2200.pgn","w")
    sys.stdout = output
    os.chdir("./lichess_db")
    filter_games(2200)
    sys.stdout = orig_stdout
    output.close()
