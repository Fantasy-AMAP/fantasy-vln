import re


def extract_substr(input_str, start_token="<answer>", end_token="</answer>"):
    start_token = re.escape(start_token)
    end_token = re.escape(end_token)
    pattern = f"{start_token}(.*?){end_token}"
    substr = re.findall(pattern, input_str, re.DOTALL)
    return ''.join(substr)

def is_valid_think_block(text: str) -> bool:
    if not isinstance(text, str):
        return False
    pattern = r'^\s*<think>[\s\S]*?</think>\s*$'
    return re.match(pattern, text) is not None
