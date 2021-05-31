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

    log_dir = "tensorboard"
    log_freq = 5  # ログの出力頻度
    save_dir = "checkpoint"
    save_freq = 1  # モデルの保存頻度

    data_dir = "dataset"

    load = False

    dir_name = "test-mtv"  # それぞれのディレクトリで保存される名前

    # 学習パラメータ
    n_step = 2
    iter_num = 10_000
    lr = 0.0005  # 学習率
    alpha_update = 0.01  # alphaの更新速度
    gamma = 0.99
    alpha = 0.0  # エントロピーの考慮具合
    target_entropy = 0.5
    rho = 0.995   # ターゲットネットワークのパラメータの移動平均の重み
    dataset_eps_size = 10000  # 保存されるエピソードの数
    learn_freq = 1  # エピソードがこの回数pushされたら学習を行う
    batch_size = 64
    bundle_size = 4

    # graph network
    global_size = 32
