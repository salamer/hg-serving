from flask import Flask
import os
from subprocess import Popen, PIPE
import psutil
import platform

# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# login(token="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")

app = Flask(__name__)


# system_prompt = "你是一个老师\n"

# tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
# model = AutoModelForCausalLM.from_pretrained(
#     'mistralai/Mistral-7B-Instruct-v0.2',
#     torch_dtype=torch.float16, # change to torch.float16 if you're using V100
#     device_map="auto",
#     use_cache=True,
# )

# conversation = [{"role": "system", "content": system_prompt }]
# while True:
#     human = input("Human: ")
#     if human.lower() == "reset":
#         conversation = [{"role": "system", "content": system_prompt }]
#         print("The chat history has been cleared!")
#         continue

#     conversation.append({"role": "user", "content": human })
#     input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

#     out_ids = model.generate(
#         input_ids=input_ids,
#         max_new_tokens=768,
#         do_sample=True,
#         top_p=0.95,
#         top_k=40,
#         temperature=0.1,
#         repetition_penalty=1.05,
#     )
#     assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
#     print("Assistant: ", assistant)
#     conversation.append({"role": "assistant", "content": assistant })


def get_all_running_processes():
    lines = []
    for process in psutil.process_iter():
        try:
            process_info = process.as_dict(attrs=["pid", "name", "username"])
            lines.append(
                f"{process_info['pid']} {process_info['name']} {process_info['username']} {process_info['cmdline'] if 'cmdline' in process_info else ''}"
            )
        except psutil.NoSuchProcess:
            pass
    return lines


def get_all_running_processes2():
    process = Popen(["ps", "aux"], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    lines = stdout.decode().split("\n")
    for line in stdout.splitlines():
        lines += line.decode()
    return lines


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/envs")
def envs():
    envs = os.environ

    envs = {k: v for k, v in envs.items()}

    return envs


@app.route("/pwd")
def pwd():
    return os.getcwd()


@app.route("/process")
def process():
    running_processes = os.popen("ps -ef").read()
    running_processes2 = os.popen("ps aux").read()
    lines = get_all_running_processes()
    return running_processes + "\n\n" + running_processes2 + "\n\n" + "\n".join(lines)


@app.route("/platform")
def platform_info():
    data = {
        "processor": platform.processor(),
        "system": platform.system(),
        "platform": platform.platform(),
        "version": platform.version(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "node": platform.node(),
        "release": platform.release(),
        "system": platform.system(),
        "uname": platform.uname(),
        "version": platform.version(),
    }
    return data


if __name__ == "__main__":
    app.run()
