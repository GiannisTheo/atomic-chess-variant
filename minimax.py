import chess
import chess.variant
import chess.pgn
import numpy as np
import os
from encoder import encode
from move_encoder import decode_move, mask_invalid,get_best_move
from copy import deepcopy

def evaluation_function(position):
    fen = position.fen().split()[0]
    rookval=4
    knightval=3
    bishopval=2
    pawnval=1
    queenval=8
    evaluation=0
    for p in fen:
        if p=='R': evaluation+=rookval
        elif p=='N': evaluation+=knightval
        elif p=='B': evaluation+=bishopval
        elif p=='Q': evaluation+=queenval
        elif p=='r': evaluation-=rookval
        elif p=='n': evaluation-=knightval
        elif p=='b': evaluation-=bishopval
        elif p=='q': evaluation-=queenval
    return evaluation

def minimax(position,player, depth, alpha, beta,counter=0):
    global bestMove
    if position.is_game_over():
        if (position.outcome().result()=='1-0'):
            return 99999
        elif (position.outcome().result()=='0-1'): 
            return -99999
        else:
            return 0
  
    if (depth==0):    
        return evaluation_function(position)


    if (player==True):    #paizei o lefkos 
        maxEval=-np.Inf  
        
        for move in position.legal_moves: 
            child=deepcopy(position) 
            child.push(move) 
            eval=minimax(child,child.turn,depth-1,alpha,beta,counter+1)
            if ((eval>maxEval) and counter==0): 
                bestMove=move 
            maxEval=max(maxEval,eval) 
            alpha=max(alpha,eval) 
            if (beta<=alpha): 
                break
        return maxEval
    else:                          
        minEval=np.Inf
        for move in position.legal_moves:
            child=deepcopy(position)
            child.push(move)
            eval=minimax(child,child.turn,depth-1,alpha,beta,counter+1)
            if ((eval<minEval) and counter==0):
                bestMove=move 
            minEval=min(minEval,eval)
            beta=min(beta,eval)
            if (beta<=alpha):
                break
        return minEval

def minimax_select(position,depth): 
    global bestMove
    bestMove=None
    minimax(position,position.turn, depth, alpha=-np.Inf, beta=np.Inf, counter = 0)
    return bestMove

if __name__=='__main__':
    position=chess.variant.AtomicBoard()
    depth=5
    move=minimax_select(position,depth)
    print(move)