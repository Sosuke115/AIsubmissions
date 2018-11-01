import chainer
DRAW = 2
import random
EMPTY=0
import environment as env
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import computational_graph as c

# Network definition
class MLP(chainer.Chain):
    #畳み込み層を追加
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer　#これは全結合層
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_units),  # Third layer
            l4=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x, t=None, train=False):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = self.l4(h)

        if train:
            return F.mean_squared_error(h,t)
        else:
            return h

    def get(self,x):
        # input x as float, output float
        return self.predict(Variable(np.array([x]).astype(np.float32).reshape(1,1))).data[0][0]


class DQNPlayer:
    def __init__(self, turn,name="DQN",e=1,dispPred=False):
        self.name=name
        self.myturn=turn
        self.model = MLP(9, 81, 9)#162は大きい 入力(状態)が1~9 出力(アクション)が1~9
        ##ここに書き加える
        gpu_device = 0
        chainer.cuda.get_device(gpu_device).use()
        self.model.to_gpu(gpu_device)
        self.xp = chainer.cuda.cupy
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.95
        self.dispPred=dispPred
        self.last_move=None
        self.last_board=None
        self.last_pred=None
        self.totalgamecount=0
        self.rwin,self.rlose,self.rdraw,self.rmiss=1,-1,0,-1.5


    def act(self,board):

        self.last_board=board.clone()
        # x=np.array([board.board],dtype=np.float32).astype(np.float32)
        x=self.xp.array([board.board],dtype=np.float32).astype(np.float32)

        pred=self.model(x)
        if self.dispPred:print(pred.data)
        self.last_pred=pred.data[0,:]
        # act=np.argmax(pred.data,axis=1)
        act=self.xp.argmax(pred.data,axis=1)

        if self.e > 0.2: #decrement epsilon over time
            self.e -= 1/(20000)
        if random.random() < self.e:
            acts=board.get_possible_pos()
            i=random.randrange(len(acts))
            act=acts[i]
        i=0
        while board.board[int(act)] != EMPTY:
            #print("Wrong Act "+str(board.board)+" with "+str(act))
            self.learn(self.last_board,act, -1, self.last_board)
            # x=np.array([board.board],dtype=np.float32).astype(np.float32)
            x=self.xp.array([board.board],dtype=np.float32).astype(np.float32)
            pred=self.model(x)
            #print(pred.data)
            # act=np.argmax(pred.data,axis=1)
            act=self.xp.argmax(pred.data,axis=1)
            i+=1
            if i>10:
                print("Exceed Pos Find"+str(board.board)+" with "+str(act))
                acts=self.last_board.get_possible_pos()
                act=acts[random.randrange(len(acts))]

        self.last_move=act
        #self.last_pred=pred.data[0,:]
        return act

    def getGameResult(self,board):
        r=0
        if self.last_move is not None:
            if board.winner is None:
                self.learn(self.last_board,self.last_move, 0, board)
                pass
            else:
                if board.board== self.last_board.board:
                    self.learn(self.last_board,self.last_move, self.rmiss, board)
                elif board.winner == self.myturn:
                    self.learn(self.last_board,self.last_move, self.rwin, board)
                elif board.winner !=DRAW:
                    self.learn(self.last_board,self.last_move, self.rlose, board)
                else:                    #DRAW
                    self.learn(self.last_board,self.last_move, self.rdraw, board)
                self.totalgamecount+=1
                self.last_move=None
                self.last_board=None
                self.last_pred=None


    def learn(self,s,a,r,fs):
        if fs.winner is not None:
            maxQnew=0
        else:
            # x=np.array([fs.board],dtype=np.float32).astype(np.float32)
            x=self.xp.array([fs.board],dtype=np.float32).astype(np.float32)
            # maxQnew=np.max(self.model(x).data[0])
            maxQnew=self.xp.max(self.model(x).data[0])
        update=r+self.gamma*maxQnew
        #print(('Prev Board:{} ,ACT:{}, Next Board:{}, Get Reward {}, Update {}').format(s.board,a,fs.board,r,update))
        #print(('PREV:{}').format(self.last_pred))
        self.last_pred[a]=update

        # x=np.array([s.board],dtype=np.float32).astype(np.float32)
        x=self.xp.array([s.board],dtype=np.float32).astype(np.float32)
        # t=np.array([self.last_pred],dtype=np.float32).astype(np.float32)
        t=self.xp.array([self.last_pred],dtype=np.float32).astype(np.float32)
        self.model.zerograds()
        loss=self.model(x,t,train=True)
        loss.backward()
        self.optimizer.update()
        #print(('Updated:{}').format(self.model(x).data))
        #print (str(s.board)+"with "+str(a)+" is updated from "+str(pQ)+" refs MAXQ="+str(maxQnew)+":"+str(r))
        #print(self.q)
