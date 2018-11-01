EMPTY=0
PLAYER_X=1
PLAYER_O=-1
MARKS={PLAYER_X:"⚫️",PLAYER_O:"⚪️",EMPTY:" "}
DRAW=2
class TTTBoard:

    def __init__(self,board=None): #board配列の初期化
        if board==None:
            self.board = []
            for i in range(9):self.board.append(EMPTY)
        else:
            self.board=board
        self.winner=None

    def get_possible_pos(self): #board配列の空いているとこのインデックスを返す
        pos=[]
        for i in range(9):
            if self.board[i]==EMPTY:
                pos.append(i)
        return pos

    def print_board(self): #boardの描画
        tempboard=[]
        for i in self.board:  #碁石の描写
            tempboard.append(MARKS[i])
        row = ' {} | {} | {} ' #ボートの描写
        # hr = '\n-----------\n'
        hr = '\nーーーーーー\n'
        print((row + hr + row + hr + row).format(*tempboard))



    def check_winner(self):
        win_cond = ((1,2,3),(4,5,6),(7,8,9),(1,4,7),(2,5,8),(3,6,9),(1,5,9),(3,5,7))#３つ並んだ時の位置
        for each in win_cond:
            if self.board[each[0]-1] == self.board[each[1]-1]  == self.board[each[2]-1]:
                if self.board[each[0]-1]!=EMPTY:
                    self.winner=self.board[each[0]-1]
                    return self.winner
        return None

    def check_draw(self):#DRAW判定
        if len(self.get_possible_pos())==0 and self.winner is None: #空いているところがゼロかつwinnerがいないなら引き分け
            self.winner=DRAW
            return DRAW
        return None

    def move(self,pos,player):#??
        if self.board[int(pos)]== EMPTY:
            self.board[int(pos)]=player
        else:
            self.winner=-1*player
        self.check_winner()
        self.check_draw()

    def clone(self):#copy??
        return TTTBoard(self.board.copy())

    def switch_player(self):#交代
        if self.player_turn == self.player_x:
            self.player_turn=self.player_o
        else:
            self.player_turn=self.player_x
