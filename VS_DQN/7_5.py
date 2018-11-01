import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import random
#7路板に拡張
#ゲームボード
class Board():
    def reset(self):
        self.board = np.array([0] * 49, dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False

    def move(self, act, turn):
        if self.board[act] == 0:
            self.board[act] = turn
            self.check_winner()
        else:
            self.winner = turn*-1
            self.missed = True
            self.done = True

    def remove(self, act):
        self.board[act] = 0
        self.winner = None
        self.done = False

    def check_winner(self):
        # win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        win_cond = []
        u = 0
        for i in range(0,7):
            win_cond.append((u,u+1,u+2,u+3,u+4))
            win_cond.append((u+1,u+2,u+3,u+4,u+5))
            win_cond.append((u+2,u+3,u+4,u+5,u+6))
            u+=7
        u = 0
        for i in range(0,7):
            win_cond.append((u,u+7,u+14,u+21,u+28))
            win_cond.append((u+7,u+14,u+21,u+28,u+35))
            win_cond.append((u+14,u+21,u+28,u+35,u+42))
            u += 1
        win_cond.append((4,10,16,22,28))
        win_cond.append((5,11,17,23,29))
        win_cond.append((11,17,23,29,35))
        win_cond.append((6,12,18,24,30))
        win_cond.append((12,18,24,30,36))
        win_cond.append((18,24,30,36,42))
        win_cond.append((13,19,25,31,37))
        win_cond.append((19,25,31,37,43))
        win_cond.append((20,26,32,38,44))

        win_cond.append((2,10,18,26,34))
        win_cond.append((1,9,17,25,33))
        win_cond.append((9,17,25,33,41))
        win_cond.append((0,8,16,24,32))
        win_cond.append((8,16,24,32,40))
        win_cond.append((16,24,32,40,48))
        win_cond.append((7,15,23,31,39))
        win_cond.append((15,23,31,39,47))
        win_cond.append((14,22,30,38,46))

        win_conditions = tuple(win_cond)
        for cond in win_conditions:
            if self.board[cond[0]] == self.board[cond[1]] == self.board[cond[2]] ==  self.board[cond[3]]  == self.board[cond[4]]:
                if self.board[cond[0]]!=0:
                    self.winner=self.board[cond[0]]
                    self.done = True
                    return


        if np.count_nonzero(self.board) == 49:
            self.winner = 0
            self.done = True

    def get_empty_pos(self):
        empties = np.where(self.board==0)[0]
        if len(empties) > 0:
            return np.random.choice(empties)
        else:
            return 0

    def show(self):
        row = " {} | {} | {} | {} | {} | {} | {} | "
        # hr = "\n-----------\n"
        hr = '\nーーーーーーーーーーーーーー\n'
        tempboard = []
        for i in self.board:
            if i == 1:
                tempboard.append("⚫️")
            elif i == -1:
                tempboard.append("⚪️")
            else:
                tempboard.append(" ")
        print((row + hr + row + hr + row + hr + row + hr + row + hr + row + hr + row).format(*tempboard))


#explorer用のランダム関数オブジェクト
class RandomActor:
    def __init__(self, board):
        self.board = board
        self.random_count = 0
    def random_action_func(self):
        self.random_count += 1
        return self.board.get_empty_pos()

#Q関数
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=98, n_hidden_channels2=196):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels2),
            l2=L.Linear(n_hidden_channels2, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))
    def __call__(self, x, test=False):
        #-1を扱うのでleaky_reluとした
        # h = F.leaky_relu(self.l0(x))
        h = F.tanh(self.l0(x))
        # h = F.leaky_relu(self.l1(h))
        h = F.tanh(self.l1(h))
        # h = F.leaky_relu(self.l2(h))
        h = F.tanh(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

# ボードの準備
b = Board()
# explorer用のランダム関数オブジェクトの準備
ra = RandomActor(b)
# 環境と行動の次元数
obs_size = 49
n_actions = 49
# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions)

#gpuに切り替え
# gpu_id = 0
# chainer.cuda.get_device(gpu_id).use()
# q_func.to_gpu(gpu_id)
#

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=600000, random_action_func=ra.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    gpu=None,minibatch_size=32,replay_start_size=20000, update_interval=1,
    target_update_interval=4000,average_q_decay=0.999, average_loss_decay=0.99)
agent_p2 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    gpu=None,minibatch_size=32,replay_start_size=20000, update_interval=1,
    target_update_interval=4000,average_q_decay=0.999, average_loss_decay=0.99)

#学習ゲーム回数
# n_episodes = 150000
n_episodes = 0
#カウンタの宣言
miss = 0
win = 0
draw = 0
a1 = 0 #それぞれのagentの勝利数
a2 = 0
#エピソードの繰り返し実行
for i in range(1, n_episodes + 1):
    b.reset()
    reward = 0
    agents = [agent_p1, agent_p2]
    turn = np.random.choice([0, 1])
    last_state = None
    while not b.done:
        #配置マス取得
        action = agents[turn].act_and_train(b.board.copy(), reward)
        #配置を実行
        b.move(action, 1)
        #配置の結果、終了時には報酬とカウンタに値をセットして学習
        if b.done == True:
            if b.winner == 1:
                reward = 1
                win += 1
                if turn == 0:
                    a1 += 1
                else:
                    a2 += 1
            elif b.winner == 0:
                draw += 1
            else:
                reward = -1
            if b.missed is True:
                miss += 1
            #エピソードを終了して学習
            agents[turn].stop_episode_and_train(b.board.copy(), reward, True)
            #相手もエピソードを終了して学習。相手のミスは勝利として学習しないように
            if agents[1 if turn == 0 else 0].last_state is not None and b.missed is False:
                #前のターンでとっておいたlast_stateをaction実行後の状態として渡す
                agents[1 if turn == 0 else 0].stop_episode_and_train(last_state, reward*-1, True)
        else:
            #学習用にターン最後の状態を退避
            last_state = b.board.copy()
            #継続のときは盤面の値を反転
            b.board = b.board * -1
            #ターンを切り替え
            turn = 1 if turn == 0 else 0

    #コンソールに進捗表示
    if i % 100 == 0:
        print("episode:", i, " / rnd:", ra.random_count, " / miss:", miss, " / win:", win, " / DQN:", a1, " / RAN:", a2, " / draw:", draw, " / statistics:", agent_p1.get_statistics(), " / epsilon:", agent_p1.explorer.epsilon)
        #カウンタの初期化
        miss = 0
        win = 0
        draw = 0
        a1 = 0
        a2 =0
        ra.random_count = 0
    if i % 10000 == 0:
        # 10000エピソードごとにモデルを保存
        agent_p1.save("7_5_150k_result_" + str(i))

print("Training finished.")
agent_p1.load("100k_result")
#人間のプレーヤー
class HumanPlayer:
    def act(self, board):
        valid = False
        while not valid:
            try:
                act = input("そーちゃんはどこに打ちますか？1-49: ")
                act = int(act)
                if act >= 1 and act <= 49 and board[act-1] == 0:
                    valid = True
                    return act-1
                else:
                    print("Invalid move")
            except Exception as e:
                print(act +  " is invalid")

class RandomPlayer:
    # def __init__(self,board,name="Random"):
        # self.name=name
    # def getGameResult(self,winner):
        # pass
    def act(self,board):
        acts = np.where(board==0)[0]
        t=random.choice(acts)
        return t#どこにおくのかを返す

class AlphaRandomPlayer:

    def act(self,board):
        acts = np.where(board==0)[0]
        for act in acts:
            # print(tempboard)
            b.move(act,-1)
            # print("error!")
            if b.winner==-1:
                b.remove(act)
                return act
            b.remove(act)


        t=random.choice(acts)
        return t#どこにおくのかを返す

class SuperAlphaRandomPlayer:

    def act(self,board):
        acts = np.where(board==0)[0]
        for act in acts:
            # print(tempboard)
            b.move(act,-1)
            # print("error!")
            if b.winner==-1:
                b.remove(act)
                return act
            else:
                acts2 = np.where(board==0)[0]
                for act2 in acts2:
                    b.move(act2,-1)
                    if b.winner==-1:
                        b.remove(act)
                        b.remove(act2)
                        return act
                    b.remove(act2)
            b.remove(act)
        t=random.choice(acts)
        return t#どこにおくのかを返す

class AlphaRandomPlayer2:
    def act(self,board):
        acts = np.where(board==0)[0]
        for act in acts:
            # print(tempboard)
            b.move(act,-1)
            # print("error!")
            if b.winner==-1:
                b.remove(act)
                return act
            b.remove(act)
        for act in acts:
            # print(tempboard)
            b.move(act,1)
            # print("error!")
            if b.winner==1:
                b.remove(act)
                return act
            b.remove(act)


        t=random.choice(acts)
        return t#どこにおくのかを返す











#検証
human_player = HumanPlayer()
# random_player = RandomPlayer()
random_player = AlphaRandomPlayer2()
# random_player = SuperAlphaRandomPlayer()
count = 0
a1 = 0
a2 = 0
draw = 0
miss = 0
while count<=1:
    count += 1
    for i in range(10):

        b.reset()
        dqn_first = np.random.choice([True, False])
        while not b.done:
            #DQN
            # print(b.board)
            if dqn_first or np.count_nonzero(b.board) > 0:
                # b.show()
                action = agent_p1.act(b.board.copy())
                b.move(action, 1)
                if b.done == True:
                    if b.winner == 1:
                        print("DQNの勝ちです")
                        a1 += 1
                    elif b.winner == 0:
                        print("Draw")
                        draw += 1
                    else:
                        print("DQN Missed")
                        miss += 1
                    agent_p1.stop_episode()
                    continue
            #人間
            b.show()
            action = human_player.act(b.board.copy())
            # action = random_player.act(b.board.copy())
            b.move(action, -1)
            if b.done == True:
                if b.winner == -1:
                    print("そーちゃんの勝ちです")
                    a2 += 1
                elif b.winner == 0:
                    draw += 1
                    print("Draw")
                agent_p1.stop_episode()

print("DQN:",a1,"random:",a2,"DRAW:",draw,"MISS:",miss)


print("Test finished.")
