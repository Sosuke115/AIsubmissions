import random
DRAW = 2

class PlayerQL:
    def __init__(self,turn,name="QL",e=0.2,alpha=0.3):
        self.name=name
        self.myturn=turn
        self.q={} #set of s,a
        self.e=e
        self.alpha=alpha
        self.gamma=0.9
        self.last_move=None
        self.last_board=None
        self.totalgamecount=0


    def policy(self,board):
        self.last_board=board.clone()
        acts=board.get_possible_pos()
        #Explore sometimes
        if random.random() < (self.e/(self.totalgamecount//10000+1)):
                i=random.randrange(len(acts))
                return acts[i]
        qs = [self.getQ(tuple(self.last_board.board),act) for act in acts]
        maxQ= max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(acts)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = acts[i]
        return acts[i]

    def getQ(self, state, act):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, act)) is None:
            self.q[(state, act)] = 1
        return self.q.get((state, act))

    def getGameResult(self,board):
        r=0
        if self.last_move is not None:
            if board.winner is None:
                self.learn(self.last_board,self.last_move, 0, board)
                pass
            else:
                if board.winner == self.myturn:
                    self.learn(self.last_board,self.last_move, 1, board)
                elif board.winner !=DRAW:
                    self.learn(self.last_board,self.last_move, -1, board)
                else:
                    self.learn(self.last_board,self.last_move, 0, board)
                self.totalgamecount+=1
                self.last_move=None
                self.last_board=None

    def learn(self,s,a,r,fs):
        pQ=self.getQ(tuple(s.board),a)
        if fs.winner is not None:
            maxQnew=0
        else:
            maxQnew=max([self.getQ(tuple(fs.board),act) for act in fs.get_possible_pos()])
        self.q[(tuple(s.board),a)]=pQ+self.alpha*((r+self.gamma*maxQnew)-pQ)
        #print (str(s.board)+"with "+str(a)+" is updated from "+str(pQ)+" refs MAXQ="+str(maxQnew)+":"+str(r))
        #print(self.q)


    def act(self,board):
        return self.policy(board)
