class Config:
    state_size = 16
    feature_size = 5
    action_size = 1
    # hidden_layers = [256, 128, 64, 32]
    hidden_layers = [48, 32, 8]
    network_in_size = 16
    network_out_size = action_size

    log_dir = "tensorboard"
    log_freq = 5  # ログの出力頻度
    save_dir = "checkpoint"
    save_freq = 1  # モデルの保存頻度

    data_dir = "dataset"

    load = False

    dir_name = "test2"  # それぞれのディレクトリで保存される名前

    lr = 0.0005  # 学習率
    alpha_update = 0.0005  # alphaの更新速度
    gamma = 0.99
    alpha = -1.0  # エントロピーの考慮具合
    target_entropy = 0
    rho = 0.995   # ターゲットネットワークのパラメータの移動平均の重み
    dataset_eps_size = 5000  # 保存されるエピソードの数
    learn_freq = 2  # エピソードがこの回数pushされたら学習を行う
    batch_size = 64
    bundle_size = 4
