import torch
from torch.utils.tensorboard import SummaryWriter
from config import Config
import json
from network import GraphSAC
import os
import shutil
from flask import make_response
import datetime


dummy_input1 = torch.randn(size=(3, 3))
dummy_input2 = torch.randn(size=(3, Config.network_in_size))
dummy_input3 = torch.randn(size=(3, Config.lstm_h_size))
dummy_input4 = torch.randn(size=(3, Config.lstm_h_size))


class Writer:
    def __init__(self):
        self.log_dir = os.path.join(Config.log_dir, Config.dir_name)
        self.save_dir = os.path.join(Config.save_dir, Config.dir_name)
        self.data_dir = os.path.join(Config.data_dir, Config.dir_name)
        self.step = 0
        self.rstep = 0
        now = datetime.datetime.now()
        self.log_name = f"{now.year}_{now.month}_{now.day}-{now.hour}:{now.minute}"
        self.log_path = os.path.join(self.log_dir, self.log_name)
        self.log_freq = Config.log_freq
        self.save_freq = Config.save_freq
        self.log_pod = log_pod()

        if Config.load is True:
            print(f"-> resume use {self.save_dir}")

            if os.path.exists(self.save_dir):
                with open(os.path.join(self.save_dir, "resume.json"), "r") as f:
                    json_dic = json.load(f)
                    self.step = json_dic["step"]
                    self.rstep = json_dic["rstep"]
                    self.log_path = json_dic["log_path"]
            else:
                print(f"-> directory {self.save_dir} not exists.")

            print(f"-> train start from {self.step} step")
            self.writer = SummaryWriter(
                log_dir=self.log_path,
            )
        else:
            if os.path.exists(self.log_path):
                shutil.rmtree(self.log_path)
            self.writer = SummaryWriter(log_dir=self.log_path)
        print(f"-> log path : {self.log_path}")

    def log_train(self, tag, value):
        self.log_pod.push(tag, value)
        if self.step % self.log_freq != 0:
            return
        self.writer.add_scalar(tag, self.log_pod.get(tag), self.step)

    def log_reward(self, value):
        self.rstep += 1
        self.log_pod.push("acum_reward", value)
        if self.rstep % self.log_freq != 0:
            return
        self.writer.add_scalar("acum_reward", self.log_pod.get("acum_reward"), self.rstep)

    def add_train_step(self):
        self.step += 1

    def add_reward_step(self):
        self.rstep += 1

    def save(self, model: GraphSAC):
        if (self.step % self.save_freq != 0):
            return
        dir_check(self.save_dir)
        torch.save(model.policy, os.path.join(self.save_dir, "policy.pth"))
        torch.save(model.q1, os.path.join(self.save_dir, "q_function_1.pth"))
        torch.save(model.q2, os.path.join(self.save_dir, "q_function_2.pth"))
        torch.save(model.q1_tar, os.path.join(self.save_dir, "q1_tar.pth"))
        torch.save(model.q2_tar, os.path.join(self.save_dir, "q2_tar.pth"))
        torch.onnx.export(
            model=model.policy,
            args=(dummy_input1, dummy_input2, dummy_input3, dummy_input4),
            f=os.path.join(self.save_dir, "policy.onnx"),
            input_names=["adj", "state", "h", "c"],
            output_names=["action", "h", "c"],
            verbose=False,
            dynamic_axes={
                'adj': {
                    0: 'node_size',
                    1: 'node_size'
                },
                'state': {
                    0: 'node_size'
                },
                'h': {
                    0: 'node_size'
                },
                'c': {
                    0: 'node_size'
                },
                'action': {
                    0: 'node_size'
                }
            }
        )

        json_dic = {
            "step": self.step,
            "rstep": self.rstep,
            "log_path": self.log_path
        }
        # 学習再開用のファイル
        with open(os.path.join(self.save_dir, "resume.json"), "w") as f:
            json.dump(json_dic, f, indent=4)

    def get_onnx_policy(self):
        # ファイルが存在しているかチェック
        if not os.path.exists(os.path.join(self.save_dir, "policy.onnx")):
            return None

        response = make_response()
        response.data = open(os.path.join(self.save_dir, "policy.onnx"), "rb").read()
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Content-Disposition'] = 'attachment; filename=policy.onnx'
        return response

    def load(self, model: GraphSAC):
        model.policy.load_state_dict(torch.load(os.path.join(self.save_dir, "policy.pth")))
        model.q1.load_state_dict(torch.load(os.path.join(self.save_dir, "q_function_1.pth")))
        model.q2.load_state_dict(torch.load(os.path.join(self.save_dir, "q_function_2.pth")))
        model.q1_tar.load_state_dict(torch.load(os.path.join(self.save_dir, "q1_tar.pthe")))
        model.q2_tar.load_state_dict(torch.load(os.path.join(self.save_dir, "q2_tar.pthe")))

        with open(os.path.join(self.save_dir, "resume.json"), "r") as f:
            json_dic = json.load(f)

        self.step = json_dic["step"]
        self.rstep = json_dic["rstep"]


def dir_check(dir):
    print(f"-> Directory check : {dir}")
    if os.path.exists(dir):
        print(f"-> {dir} exists")
    else:
        print(f"-> {dir} not exists")
        os.makedirs(dir)
        print(f"-> make {dir}")


class log_pod:
    def __init__(self):
        self.dict = {}

    def push(self, tag, value):
        if tag in self.dict:
            self.dict[tag].append(value)
        else:
            self.dict[tag] = [value]

    def get(self, tag):
        value = sum(self.dict[tag]) / len(self.dict[tag])
        self.dict[tag] = []
        return value
