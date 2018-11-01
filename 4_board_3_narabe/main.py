import agents
import environment as env
import pickle
import _pickle
# import dill
EMPTY=0
PLAYER_X=1
PLAYER_O=-1
MARKS={PLAYER_X:"⚫️",PLAYER_O:"⚪️",EMPTY:" "}
DRAW=2


def train():
    p1=agents.PlayerHuman(PLAYER_X)
    # p1=agents.PlayerQL(PLAYER_X)
    # p1 = agents.PlayerMC(PLAYER_X)
    # p1 = agents.DQNPlayer(PLAYER_X)
    # p1=agents.PlayerAlphaRandom(PLAYER_X)
    # p2=agents.PlayerBetaRandom(PLAYER_O)
    p2 = agents.PlayerMC(PLAYER_O)
    # p2=agents.PlayerRandom(PLAYER_O)
    # p2=agents.PlayerQL(PLAYER_O)
    # p2=agents.PlayerQL(PLAYER_O,"QL2")
    # p2 = agents.DQNPlayer(PLAYER_O)
    # p2=agents.PlayerAlphaRandom(PLAYER_O)
    # with open('pdq_vs_pdq_10000_X.pickle', mode='rb') as f:
        # p1 = pickle.load(f)
    # p2.e = 0
    # with open('sample.pickle', mode='rb') as f:
        # p2 = pickle.load(f)
    # p2.e = 0
    game=agents.TTT_GameOrganizer(p1,p2)
    # game=agents.TTT_GameOrganizer(p1,p2,1000,False,False,10) #最初が対戦回数、次が表示
    game.progress()

    with open('saisyu_mc_3_1000.pickle', mode='wb') as f:
        pickle.dump(p1, f)
    # p2.model.to_cpu()
    # _pickle.dump(p2, open("model.pkl", "wb"), -1)
    # with open('zikken2.pickle', mode='wb') as f:
        # pickle.dump(p2, f)


# train()



def test():
    p1=agents.PlayerHuman(PLAYER_X,"そーちゃん")
    # p2=agents.PlayerBetaRandom(PLAYER_O)
    # p1=agents.PlayerQL(PLAYER_X,"QL1")
    # p2=agents.PlayerAlphaRandom(PLAYER_O)
    # p2=agents.PlayerRandom(PLAYER_O)
    # p2=agents.PlayerQL(PLAYER_O)
    p2 = agents.PlayerMC(PLAYER_O)
    # with open('pdq_vs_pdq_10000_X.pickle', mode='rb') as f:
        # p1 = pickle.load(f)
    # p1.e = 0
    # with open('saisyu_ql_3_100000.pickle', mode='rb') as f:
        # p2 = pickle.load(f)
    # p1.e = 0
    # p2=agents.PlayerQL(PLAYER_O,"QL2")
    # p2=agents.PlayerAlphaRandom(PLAYER_O)
    game=agents.TTT_GameOrganizer(p1,p2)
    # game=agents.TTT_GameOrganizer(p1,p2,1000,False,False,100) #最初が対戦回数、次が表示
    game.progress()

test()
