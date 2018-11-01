import random

class PlayerAlphaRandom:


    def __init__(self,turn,name="AlphaRandom"):
        self.name=name
        self.myturn=turn

    def getGameResult(self,winner):
        pass

    def act(self,board):
        acts=board.get_possible_pos()
        #see only next winnable act
        for act in acts:
            tempboard=board.clone()
            tempboard.move(act,self.myturn)
            # check if win
            if tempboard.winner==self.myturn:
                #print ("Check mate")
                return act
        i=random.randrange(len(acts))
        return acts[i]
