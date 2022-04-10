from flask import Flask, jsonify, request
import json
from agent import Agent, GraphDataset
from typing import Dict, Any
import torch
import torch.optim as optim
import random
from network import GraphSAC
from trainer import train
from config import Config
from utils import Writer, Timewatch
import logging
from collections import deque
from threading import Thread


random.seed(2)
torch.manual_seed(2)

# ロガーの取得
werkzeug_logger = logging.getLogger("werkzeug")
# レベルの変更
werkzeug_logger.setLevel(logging.ERROR)

app = Flask(__name__)

agents: Dict[int, Agent] = {}

model = GraphSAC()
value_optim = optim.Adam([
    # {"params": model.q.parameters()}
    {"params": model.q1.parameters()},
    {"params": model.q2.parameters()}
    # {"params": model.v.parameters()}
], Config.lr)
policy_optim = optim.Adam(model.policy.parameters(), Config.lr)
alpha_optim = optim.Adam([model.alpha], Config.alpha_update)

writer = Writer()
watch = Timewatch(Config.log_smoothing)

writer.save(model)
train_q = deque([], maxlen=10)


def train_worker(q):
    while True:
        if len(q) == 0:
            continue
        batches = q.popleft()
        if writer.step < Config.iter_num:
            policy_loss, value_loss = train(model, batches, writer, value_optim, policy_optim, alpha_optim)
            print(f"\r [{writer.step} step : {watch.call():.2g}s/iter] policy_loss:{policy_loss:.5g},  value_loss:{value_loss:.5g}          ", end="")
        else:
            print("End Learn")
            # ログを何もしない関数に書き換える
            break


def train_func(batches):
    train_q.append(batches)


dataset = GraphDataset(train_func=train_func,
                       log_reward=writer.log_reward)

# trainスレッドの作成、常駐化
thread = Thread(target=train_worker, args=(train_q,))
thread.setDaemon(True)
thread.start()


@app.route("/", methods=["GET"])
def hello_check() -> str:
    out_text = ""
    for key, value in agents.items():
        out_text += "id :" + str(key) + "\n" + str(value)

    return out_text


# @app.route("/", methods=["POST"])
# def init_academy():
#     return "hoge"


@app.route("/new", methods=["POST"])
def new_agent() -> Any:
    req_data = json.loads(request.data)
    print(req_data)
    id = random.randint(0, 2**20)
    agent = Agent(
        adj_mat=torch.Tensor(req_data["Graph"]).reshape(req_data["NodeSize"], -1),
        dataset=dataset
    )

    agents[id] = agent

    res_data: Dict[str, int] = {"Id": id}
    return jsonify(res_data)  # type: ignore


@app.route("/push/<int:id>", methods=["POST"])
def push_data(id):
    # assert(id in agents.keys())
    # print(f"ID: {id} not found")
    req_data = json.loads(request.data)
    agent = agents[id]
    agent.push_data(
        state=torch.Tensor(req_data["State"]).reshape(agent.node_size, -1),
        action=torch.Tensor(req_data["Action"]).reshape(agent.node_size, -1),
        reward=req_data["Reward"],
        done=req_data["Done"]
    )

    return "ok"


@app.route("/onnx", methods=["GET"])
def get_onnx_policy():
    return writer.get_onnx_policy()


@app.route("/save", methods=["GET"])
def save():
    writer.save(model)
    return "ok"


@app.route("/stop/<int:id>", methods=["GET"])
def stop_agent(id):
    assert(id in agents.keys())
    agents.pop(id)
    return "ok"


@app.route("/step", methods=["GET"])
def get_step():
    return str(1 + (writer.step // Config.save_freq))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
