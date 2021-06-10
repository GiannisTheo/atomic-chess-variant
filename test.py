import chess
import chess.variant
import chess.pgn

board=chess.variant.AtomicBoard()

print(board.fen())
print(board)
Nf3 = chess.Move.from_uci("g1f3")
board.push(Nf3)
e5=chess.Move.from_uci("e7e5")
board.push(e5)
print(board)