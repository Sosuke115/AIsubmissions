import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import random

#ゲームボード
class Board():
    def reset(self):
        self.board = np.array([0] * 9, dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False

    def move(self, act, turn):
        if self.board[act] == 0:#board[act]が埋まっていなかったら
            self.board[act] = turn#ソノターンの人の石としてマスをうめる(+-1?)
            self.check_winner()#勝ってるか判定
        else:
            self.winner = turn*-1
            self.missed = True#同じ場所にうってるのでミス
            self.done = True

    def remove(self, act):
        self.board[act] = 0
        self.winner = None
        self.done = False

    def check_winner(self):
        win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for cond in win_conditions:
            if self.board[cond[0]] == self.board[cond[1]] == self.board[cond[2]]:
                if self.board[cond[0]]!=0:
                    self.winner=self.board[cond[0]]
                    self.done = True
                    return
        if np.count_nonzero(self.board) == 9:
            self.winner = 0
            self.done = True

    def get_empty_pos(self):#空いている位置を探す
        empties = np.where(self.board==0)[0]
        if len(empties) > 0:
            return np.random.choice(empties)
        else:
            return 0

    def show(self):
        row = " {} | {} | {} "
        hr = "\n-----------\n"
        tempboard = []
        for i in self.board:
            if i == 1:
                tempboard.append("⚫️")
            elif i == -1:
                tempboard.append("⚪️")
            else:
                tempboard.append(" ")
        print((row + hr + row + hr + row).format(*tempboard))

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
    def __init__(self, obs_size, n_actions, n_hidden_channels=81):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))
    def __call__(self, x, test=False):
        #-1を扱うのでleaky_reluとした
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

# ボードの準備
b = Board()
# explorer用のランダム関数オブジェクトの準備
ra = RandomActor(b)
# 環境と行動の次元数
obs_size = 9
n_actions = 9
# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions)

#gpuに切り替え
# gpu_id = 0
# chainer.cuda.get_device(gpu_id).use()
# q_func.to_gpu(gpu_id)
#
class RandomPlayer:
    # def __init__(self,board,name="Random"):
        # self.name=name

    # def getGameResult(self,winner):
        # pass

    def act(self,board):
        acts = np.where(board==0)[0]
        # acts=board.get_possible_pos()
        #see only next winnable act
        # for act in acts:
            # tempboard=b.copy()
            # tempboard.move(act,1)
            # check if win
            # if tempboard.winner==self.myturn:
                #print ("Check mate")
                # return act
        i=random.randrange(len(acts))
        return acts[i]#どこにおくのかを返す

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

class BetaRandomPlayer:
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

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=ra.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    gpu=None,replay_start_size=500, update_interval=1,
    target_update_interval=100)
agent_p2 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    gpu=None,replay_start_size=500, update_interval=1,
    target_update_interval=100)

#学習ゲーム回数
# n_episodes = 50000
n_episodes = 0
#カウンタの宣言
miss = 0 #同じとこにおく回数
win = 0
draw = 0
a1 = 0 #それぞれのagentの勝利数
a2 = 0
#エピソードの繰り返し実行
for i in range(1, n_episodes + 1):
    b.reset()#ボードのリセット
    reward = 0#報酬
    agent_p2 = RandomPlayer(1)#2をランダム君に
    agents = [agent_p1, agent_p2]
    turn = np.random.choice([0, 1])#最初のターン

    last_state = None#最後の状態
    while not b.done:#boardが埋められてない状態
        #配置マス取得
        if turn == 0:
            action = agents[turn].act_and_train(b.board.copy(), reward)
        else:
            action = agents[turn].act(b.board.copy())

        #配置を実行
        b.move(action, 1)
        #配置の結果、終了時には報酬とカウンタに値をセットして学習
        if b.done == True:#終了している
            if b.winner == 1:#自分のかち
                reward = 1
                win += 1
                if turn == 0:
                    a1 += 1
                else:
                    a2 += 1


            elif b.winner == 0:#引き分け
                draw += 1
            else:#-1なら相手の勝？
                reward = -1
            if b.missed is True:
                miss += 1
            #エピソードを終了して学習
            if turn == 0:
                agents[turn].stop_episode_and_train(b.board.copy(), reward, True)


            # agents[turn].stop_episode_and_train(b.board.copy(), reward, True)
            #相手もエピソードを終了して学習。相手のミスは勝利として学習しないように
            # if agents[1 if turn == 0 else 0].last_state is not None and b.missed is False:
                #前のターンでとっておいたlast_stateをaction実行後の状態として渡す
                # agents[1 if turn == 0 else 0].stop_episode_and_train(last_state, reward*-1, True)
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
        a2 = 0
        ra.random_count = 0
    if i % 10000 == 0:
        # 10000エピソードごとにモデルを保存
        agent_p1.save("saisyu_ql_3_50000_" + str(i))

print("Training finished.")
agent_p1.load("saisyu_ql_3_50000_50000")

#人間のプレーヤー
class HumanPlayer:
    def act(self, board):
        valid = False
        while not valid:
            try:
                act = input("Please enter 1-9: ")
                act = int(act)
                if act >= 1 and act <= 9 and board[act-1] == 0:
                    valid = True
                    return act-1
                else:
                    print("Invalid move")
            except Exception as e:
                print(act +  " is invalid")

#検証
human_player = HumanPlayer()
# random_player = RandomPlayer()
# random_player = AlphaRandomPlayer()
# random_player = BetaRandomPlayer()
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
                b.show()
                action = agent_p1.act(b.board.copy())
                b.move(action, 1)
                if b.done == True:
                    if b.winner == 1:
                        print("DQN Win")
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
                    print("HUMAN Win")
                    a2 += 1
                elif b.winner == 0:
                    draw += 1
                    print("Draw")
                agent_p1.stop_episode()

print("DQN:",a1,"random:",a2,"DRAW:",draw,"MISS:",miss)

print("Test finished.")
