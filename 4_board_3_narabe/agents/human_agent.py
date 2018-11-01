
DRAW=2

class PlayerHuman:
    def __init__(self,turn,name="Human"):
        self.name=name
        self.myturn=turn#先攻か後攻か　　　　　　　　　　　　　

    def act(self,board):
        valid = False
        while not valid:#valid=Trueで抜け出せる
            try:
                act = input("Where would you like to place " + str(self.myturn) + " (1-9)? ")#数字を読み込む
                act = int(act)
                #if act >= 1 and act <= 9 and board.board[act-1]==EMPTY:
                if act >= 1 and act <= 9:
                    valid=True
                    return act-1#インデックスを返す
                else:
                    print ("That is not a valid move! Please try again.")
            except Exception as e:
                    print (act +  "is not a valid move! Please try again.")
        return act

    def getGameResult(self,board):
        if board.winner is not None and board.winner!=self.myturn and board.winner!=DRAW:#負けた
            print("I lost...")
