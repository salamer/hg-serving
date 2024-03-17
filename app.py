from flask import Flask
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

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()