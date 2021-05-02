class Config:
    state_size = 16
    feature_size = 5
    action_size = 1
    action_num = 3
    # hidden_layers = [256, 128, 64, 32]
    hidden_layers = [128, 64, 16, 4]
    network_in_size = 16
    network_out_size = action_num
    dropout_p = 0.1

    # Utillパラメータ
    log_dir = "tensorboard"
    log_freq = 5  # ログの出力頻度
    save_dir = "checkpoint"
    save_freq = 1  # モデルの保存頻度

    data_dir = "dataset"

    load = False

    dir_name = "r2d2-test"  # それぞれのディレクトリで保存される名前

    # 強化学習パラメータ
    gamma = 0.99
    # 学習パラメータ
    lr = 0.0005                 # 学習率
    # r2d2パラメータ
    n_step = 4                  # multi-step learningの考慮ステップ数 最小は1
    burn_in = 16                # burn-inを行う期間
    seq_in = 16                 # burn-inの後に入力する期間
    pick_out_range = burn_in + seq_in
    rnn = True
    lstm_h_size = 16

    # sacパラメータ
    alpha_update = 0.01         # alphaの更新速度
    alpha = 0.0                 # 初期エントロピーの考慮具合
    target_entropy = 0.5
    rho = 0.995                 # ターゲットネットワークのパラメータの移動平均の重み
    dataset_eps_size = 10000    # 保存されるエピソードの数
    learn_freq = 1              # エピソードがこの回数pushされたら学習を行う
    batch_size = 64             # 一つのエピソードから取り出すバッチのサイズ
    bundle_size = 4             # 一回の学習で取り出すエピソードの数

    # graph network
    global_size = 16
