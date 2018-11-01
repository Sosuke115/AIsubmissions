4_board_3_narabe
5_board_4_narabe
7_board_5_narabe
VS_DQN
の４つのディレクトリが存在します。それぞれについてディレクトリ内でできることを以下に述べます。

・4_board_3_narabeについて
題名通り、4路盤3目並べにてAIとの対戦、AI同士の対戦が可能です。
デフォルトではシェルコマンドで
python main.py
と打つことで、3目並べモンテカルロAIと対戦できます。対戦相手の変更にはmain.pyにて、test関数内のp1、p2を有効無効にすることで変更可能です。例えば対戦をモンテカルロvsAlphaRandomにしたい時は

p1=agents.PlayerHuman(PLAYER_X,"そーちゃん")を
コメントアウトし、p1=agents.PlayerAlphaRandom(PLAYER_X)のコメントアウトをとります。その後、同様にシェルコマンドで
python main.py
と打つことで、モンテカルロvsAlphaRandomの対戦が見られます。
また、Q学習との対戦の際には途中にコメントアウトされている
with open('saisyu_ql_3_100000.pickle', mode='rb') as f:
      p2 = pickle.load(f)
  p2.e = 0
を有効にすることで、学習済みQ学習をプレイヤー2（p2）に設定できます。

注意点として、このディレクトリではDQNは実装されていますが、８班で工夫して実装したのはVS_DQNディレクトリ内のDQNエージェントです。

・5_board_4_narabeについて
題名通り、5路盤4目並べにてAIとの対戦、AI同士の対戦が可能です。
動かし方は4_board_3_narabeと同じですので説明は省略します。

・7_board_5_narabeについて
題名通り、7路盤5目並べにてAIとの対戦、AI同士の対戦が可能です。
動かし方は4_board_3_narabeと同じですので説明は省略します。

・VS_DQNについて
このディレクトリ内では、3,4,5目並べでDQNとの対戦、DQNとランダムの対戦が可能です。デフォルトでは
python 7_5.py
と打つことで学習済みDQNと対戦可能です。
ランダムとDQNの対戦は7-5.py内で367行目の
action = random_player.act(b.board.copy())
を有効にし、366行目をコメントアウトします。その後331〜333行目から対戦させたいランダムプレイヤーを選び、同様に実行することで対戦が見られます。
3,4目並べを実行したい時は、5_4.pyや4_3.pyにて同様な操作をしてください。
