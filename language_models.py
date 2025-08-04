import os
import requests
import json

API_BASE_URL = "https://api.360.cn/v1"

def get_360_api_response(prompt, model, is_judge=False, stream=False):
    url = f"{API_BASE_URL}/chat/completions"
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API_KEY unset")
        return {
            "response": "API Key 未设置",
            "cost": 0,
            "length_cutoff": False,
            "safety_cutoff": False,
            "api_error": True
        }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": 0.9,
        "max_tokens": 16384
    }

    try:
        if stream:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180, stream=True)
            response.raise_for_status()
            full_content = ""
            finish_reason = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[len('data: '):]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(json_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content_part = delta.get("content", "")
                            full_content += content_part
                            if chunk.get("choices", [{}])[0].get("finish_reason"):
                                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                        except json.JSONDecodeError:
                            print(f"无法解析的 JSON 数据块: {json_str}")
                            continue
            if is_judge:
                return full_content
            return {
                "response": full_content,
                "cost": 0,
                "length_cutoff": finish_reason == "length",
                "safety_cutoff": finish_reason == "content_filter",
                "api_error": False
            }
        else:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
            response.raise_for_status()
            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            finish_reason = response_data.get("choices", [{}])[0].get("finish_reason", "")
            if is_judge:
                return content
            return {
                "response": content,
                "cost": 0,
                "length_cutoff": finish_reason == "length",
                "safety_cutoff": finish_reason == "content_filter",
                "api_error": False
            }
    except requests.exceptions.RequestException as e:
        print(f"API 请求时发生错误: {e}")
        if 'response' in locals() and hasattr(response, "text"):
            print(response.text)
        return {
            "response": f"API 错误: {e}",
            "cost": 0,
            "length_cutoff": False,
            "safety_cutoff": False,
            "api_error": True
        }

def get_llm_response(prompt, model, judge=False, stream=False):
    return get_360_api_response(prompt, model, is_judge=judge, stream=stream)
