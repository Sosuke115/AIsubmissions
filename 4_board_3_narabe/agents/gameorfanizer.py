import environment as env
DRAW=2
import random
class TTT_GameOrganizer: #ゲームの進行役

    act_turn=0
    winner=None

    def __init__(self,px,po,nplay=1,showBoard=True,showResult=True,stat=100):#(p1,p2,プレイ数,ボードをみるか,結果表示するか,何回ごとに)
        self.player_x=px
        self.player_o=po
        self.nwon={px.myturn:0,po.myturn:0,DRAW:0}
        self.nplay=nplay
        self.players=(self.player_x,self.player_o)
        self.board=None
        self.disp=showBoard
        self.showResult=showResult
        self.player_turn=self.players[random.randrange(2)]#最初のターンはランダム
        self.nplayed=0
        self.stat=stat

    def progress(self):
        while self.nplayed<self.nplay:
            self.board=env.TTTBoard()
            while self.board.winner==None:
                if self.disp:print(self.player_turn.name+"さんの番です")
                act=self.player_turn.act(self.board)
                self.board.move(act,self.player_turn.myturn)
                if self.disp:self.board.print_board()

                if self.board.winner != None:#勝者がいたら
                    # notice every player that game ends
                    for i in self.players:
                        i.getGameResult(self.board)
                    if self.board.winner == DRAW:
                        if self.showResult:print ("引き分け")
                    elif self.board.winner == self.player_turn.myturn:
                        out = self.player_turn.name+"さんの勝ちです"
                        if self.showResult: print(out)
                    else:
                        print ("打てません！")
                    self.nwon[self.board.winner]+=1
                else:#勝者がいなかったら
                    self.switch_player()
                    #Notice other player that the game is going
                    self.player_turn.getGameResult(self.board)

            self.nplayed+=1
            if self.nplayed%self.stat==0 or self.nplayed==self.nplay:
                print(self.player_x.name+":"+str(self.nwon[self.player_x.myturn])+","+self.player_o.name+":"+str(self.nwon[self.player_o.myturn])+",DRAW:"+str(self.nwon[DRAW]))


    def switch_player(self):
        if self.player_turn == self.player_x:
            self.player_turn=self.player_o
        else:
            self.player_turn=self.player_x
