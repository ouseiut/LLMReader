# main
import argparse
import PyPDF2
import requests
import sys
from openai import OpenAI
from PyPDF2 import PdfReader


configs = {
    "deepseek":{
        'url':"https://api.deepseek.com",
        'key':"",
        'models':[
            'deepseek-chat',
            'deepseek-reasoner'
        ]
    },
    "zju":{
        'url':"https://chat.zju.edu.cn/api/ai/v1",
        'key':"",
        'models':[
            'deepseek-r1-671b',
            'deepseek-v3-671b',
            'deepseek-r1-distill-qwen',
            'qwen2.5-instruct',
            'bge-m3',
            'bge-reranker-v2-m3'
        ]
    },
    "jlu":{
        'url':"https://deepseek.jlu.edu.cn/api/ai/v1",
        'key':"",
        'models':[
            'DeepSeek-R1'
        ]
    },
    "vapi":{
        'url':"https://api.gpt.ge/v1/",
        'key':"",
        'models':[
            'gpt-4o-all',
            'gpt-4-all',
            'gpt-4o-2024-08-06'
            
        ]
    }
}

import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_map):
    # 使用最小堆构建哈夫曼树
    heap = [HuffmanNode(char, freq) for char, freq in freq_map.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        # 取出频率最小的两个节点
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        # 创建一个新节点，频率为两个节点的和
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    # 返回哈夫曼树的根节点
    return heap[0]

def build_huffman_code_table(root, code="", code_table=None):
    if code_table is None:
        code_table = {}
    if root:
        if root.char is not None:
            code_table[root.char] = code
        build_huffman_code_table(root.left, code + "0", code_table)
        build_huffman_code_table(root.right, code + "1", code_table)
    return code_table

def huffman_encode(text):
    if not text:
        return "", {}

    # 统计字符频率
    freq_map = Counter(text)
    # 构建哈夫曼树
    huffman_tree = build_huffman_tree(freq_map)
    # 生成编码表
    code_table = build_huffman_code_table(huffman_tree)
    # 编码文本
    encoded_text = "".join(code_table[char] for char in text)
    return encoded_text, code_table

def compress_text(text):
    if not text:
        return b"", {}

    # 哈夫曼编码
    encoded_text, code_table = huffman_encode(text)
    # 将二进制字符串转换为字节
    padding = 8 - len(encoded_text) % 8
    encoded_text += "0" * padding  # 填充到 8 的倍数
    byte_array = bytearray()
    for i in range(0, len(encoded_text), 8):
        byte = encoded_text[i:i + 8]
        byte_array.append(int(byte, 2))
    # 返回压缩后的字节数据和编码表
    return bytes(byte_array), code_table, padding

def decompress_text(compressed_data, code_table, padding):
    if not compressed_data:
        return ""

    # 将字节转换为二进制字符串
    binary_str = ""
    for byte in compressed_data:
        binary_str += f"{byte:08b}"
    # 去除填充的 0
    binary_str = binary_str[:-padding]
    # 构建反向编码表（从编码到字符）
    reverse_code_table = {code: char for char, code in code_table.items()}
    # 解码文本
    current_code = ""
    decoded_text = ""
    for bit in binary_str:
        current_code += bit
        if current_code in reverse_code_table:
            decoded_text += reverse_code_table[current_code]
            current_code = ""
    return decoded_text

def set_configs():
    global configs
    return

def read_pdf(file_name):

    reader = PdfReader(file_name)
    number_of_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
class Chatter:
    def __init__(self, curr_cfg, model) -> None:
        print("System Init.")
        self.curr_cfg = curr_cfg
        self.client = OpenAI(api_key=curr_cfg['key'], base_url=curr_cfg['url'])
        self.model = model

        pass
    
    def unlimited_chat(self,clip = -1):

        messages = [
        {"role": "system", "content": "你是一个见多识广的计算机领域专家，尤其是在机器学习，大模型和强化学习领域。你应当尽可能响应用户的所有需求。请不要返回任何emoji表情。"},
        {"role": "user", "content": "Hello"},
        ]
        response = self.client.chat.completions.create(
        model=self.curr_cfg['models'][self.model],
        messages=messages,
        stream=False,
        # max_tokens=int(1e9)
        )
        print(response.choices[0].message.content)
        print("Done.")
        while True:
            messages = [
            {"role": "system", "content": "你是一个见多识广的计算机领域专家，尤其是在机器学习，大模型和强化学习领域。接下来请响应用户的所有需求，请不要返回任何emoji表情。"},
            ]
            user_input = input("waiting for user input:")
            if user_input == "###END###": break
            if user_input.endswith("#pdf#"):
                task_name = input("Please input the name of pdf:")
                user_input.replace("#pdf#", "")
                user_input += read_pdf(task_name)
                # user_input = user_input[]
            
            if clip > 0:
                clices = []
                user_input = f"请注意，接下来的输入很长。由于token数量限制，我不得不分片给你.被分段的内容将以***+++***标志。***+++***" + user_input

                clices = [user_input[i:i+clip] + "***+++***" for i in range(0, len(user_input), clip)]
                clices.append("所有分片发送结束。")
                user_inputs = clices
            else:
                user_inputs = [user_input]
            
            for user_input in user_inputs:
                messages.append({"role": "user", "content": str(user_input)})
            response = self.client.chat.completions.create(
                model=self.curr_cfg['models'][self.model],
                messages=messages
            )
            print(f"{self.curr_cfg['models'][self.model]}: {response.choices[0].message.content}")

            print(f"------ [ i/o tokens: {response.usage.prompt_tokens} / {response.usage.completion_tokens} ] ------")
            price_i = 1 * response.usage.prompt_tokens/(int(1e6))
            price_o = 4 * response.usage.prompt_tokens/(int(1e6))
            print(f"------ [ Costs (Discounted): {price_i:.9f} / {price_o:.9f} ] ------")
        print("Exited.")
        return





def main(args):
    
    global configs
    curr_cfg = configs[args.server]


    chatter = Chatter( curr_cfg, args.model )    
    chatter.unlimited_chat()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="vapi", choices=[
        "deepseek",
        'zju',
        'jlu'
    ])

    parser.add_argument("--model", type=int, default=0)

    args = parser.parse_args()

    
    main(args)


    sys.exit(0)
