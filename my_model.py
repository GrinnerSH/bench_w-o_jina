import os
import json
import re
import requests
from typing import Dict, Any, List

QWEN_API_URL = "https://api.360.cn/v1/chat/completions"
QWEN_API_KEY = os.getenv("API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


MASTER_SYSTEM_PROMPT = """
# 角色与目标
你是一个专家级的AI研究智能体。你的唯一目标是：通过逻辑严谨的步骤分解复杂问题，并利用网页搜索工具找到最权威、最准确的答案。你必须做到思路清晰、回答精确，并为你提供的每一个关键信息注明来源URL。


# 核心工作循环: ReAct-RRSM
你将遵循一个严格的“思考->行动->观察->反思”循环来解决问题。


# 核心原则：事实核查与交叉验证 (Fact-Checking & Cross-Validation)
你必须像一个严谨的研究员一样工作。绝对不允许仅仅依赖搜索结果的摘要（description）就得出结论。首先，请深入分析用户的问题，并动用你自己的知识库进行尝试性解答。如果你确保你的知识**足够且正确**，请直接整理思路并给出当前阶段答案。如果你的知识**不足**、**可能已过时**或**需要事实核查**，则启动下面的研究流程。
你的信息处理流程必须遵循以下步骤：
1.  **初步搜索 (Initial Search)**: 使用 `jina_search` 获取一个包含多个来源的列表。
2.  **来源研判 (Source Analysis)**: 在你的 `Thought` 中，分析并比较返回的所有搜索结果。评估每个来源的权威性（例如，官方政府网站 > 知名新闻机构 > 个人博客）。识别信息中的一致性、差异或矛盾。
3.  **假设与核查 (Hypothesize & Verify)**:
    - 基于你的研判，选择 **最权威的一到两个URL**。
    - **必须** 接着使用 `jina_reader` 工具进入这些URL，读取其 **完整内容** 来寻找并核实具体的关键信息（如具体的数字、日期、官方声明等）。
    - **永远不要** 将搜索摘要中的信息直接当作最终事实。摘要可能不完整或有误导性。
4.  **得出结论 (Conclude)**: 只有在你通过 `jina_reader` 核实了信息之后，才能在你的思考过程中确认这个事实，并进行下一步的计算或推理。如果信息相互矛盾，你需要指出矛盾点，并可能需要进行新一轮的搜索来解决。


# 思考 (Thought)
在每次输出`Action`之前，你必须先进行一步`Thought`。在`Thought`中，你需要：
1. 回顾你当前的总体目标和正在解决的子问题。
2. 分析上一步的`Observation`（观察结果），并遵循“事实核查与交叉验证”原则。
3. 清晰地阐述你下一步行动的理由。你的思考过程应该展现出逻辑性、计划性和批判性思维。


# 行动 (Action) 规范
你的所有“行动”都必须以一个严格的JSON格式代码块输出。这是强制性规定。你的行动必须是以下三种之一：

1.  **搜索网页**:
    ```json
    {
        "tool_name": "jina_search",
        "parameters": {
            "query": "<这里是高度具体的搜索关键词>"
        }
    }
    ```

2.  **阅读网页内容**:
    ```json
    {
        "tool_name": "jina_reader",
        "parameters": {
            "url": "<这里是要读取的有效URL>"
        }
    }
    ```

3.  **提供最终答案**:
    ```json
    {
        "tool_name": "final_answer",
        "parameters": {
            "answer": "<这里是问题的最终答案>",
            "sources": ["<来源URL 1>", "<来源URL 2>"]
        }
    }
    ```

# 输出格式
你必须严格遵循以下格式进行输出，不得有任何偏差：
Thought: <这里是你的思考过程>
Action:
```json
<这里是你的JSON格式行动指令>
```"""


def _call_qwen_llm(messages: List[Dict[str, Any]], stream: bool = True) -> requests.Response:
    """Helper function to make a call to the Qwen LLM API."""
    if not QWEN_API_KEY:
        raise ValueError("SHANLING_API_KEY environment variable not set.")

    payload = {
        "model": "alibaba/qwen3-32b",
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.9,
        "stream": stream
    }
    headers = {
        'Authorization': f'Bearer {QWEN_API_KEY}',
        'Content-Type': 'application/json'
    }
    return requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=300, stream=stream)


def _get_streamed_content(response: requests.Response) -> str:
    """Aggregates content from a streaming LLM response."""
    full_content = ""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode('utf-8')
        if decoded.startswith('data:'):
            json_str = decoded[len('data:'):].strip()
            if json_str == '[DONE]':
                break
            try:
                chunk = json.loads(json_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    full_content += delta["content"]
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON chunk: {json_str}")
    return full_content


def run_jina_search(query: str) -> str:
    """
    当需要获取实时信息、查找特定主题资料时, 使用此工具进行网络搜索。
    """
    if not JINA_API_KEY:
        return json.dumps({"error": "JINA_API_KEY environment variable not set."})
    try:
        search_url = f"https://s.jina.ai/{query}"
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Accept": "application/json"
        }
        response = requests.get(search_url, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data:
            return "No search results found."

        results = []
        for i, item in enumerate(data[:5], 1):
            title = item.get('title', 'No Title')
            url = item.get('url', 'No URL')
            desc = item.get('description', 'No Description')
            results.append(f"{i}. Title: {title}\n   URL: {url}\n   Description: {desc}")
        return "\n\n".join(results)
    except requests.RequestException as e:
        return json.dumps({"error": f"Jina search failed: {e}"})


def run_jina_reader(url: str, question: str) -> str:
    """
    阅读并提取指定URL网页的核心文本内容。
    """
    if not JINA_API_KEY:
        return json.dumps({"error": "JINA_API_KEY environment variable not set."})
    try:
        reader_url = f"https://r.jina.ai/{url}"
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Accept": "application/json"
        }
        response = requests.get(reader_url, headers=headers, timeout=120)
        response.raise_for_status()
        raw = response.json().get("data", {}).get("content", "")
        if not raw:
            return "Could not retrieve content from the URL."

        refinement_prompt = f"""
        # 指令: 信息提取与精炼
        你是一个信息提取引擎。你的任务是从以下提供的原始网页文本中, 根据当前需要回答的子问题, 提取出所有相关的句子、段落和事实数据。
        当前子问题: "{question}"
        来源URL: "{url}"
        原始文本:
        ---
        {raw[:20000]}
        ---
        请忽略所有的HTML标签、导航链接、广告、版权声明、用户评论区以及其他与回答子问题无关的样板文字。将提取出的干净、相关的核心信息, 以一个清晰的文本块形式呈现。
        """
        messages = [{"role": "user", "content": refinement_prompt}]
        refine_resp = _call_qwen_llm(messages, stream=True)
        refine_resp.raise_for_status()
        return _get_streamed_content(refine_resp) or "Content refinement failed."
    except requests.RequestException as e:
        return json.dumps({"error": f"Jina reader failed for url {url}: {e}"})


class RRSMAgent:
    # def __init__(self, max_turns: int = 10):
    def __init__(self, max_turns: int = 20):
        self.llm_call = _call_qwen_llm
        self.tools = {
            "jina_search": run_jina_search,
            "jina_reader": run_jina_reader,
        }
        self.max_turns = max_turns
        self.scratchpad: Dict[str, Any] = {
            "overall_objective": "",
            "reasoning_plan": [],
            "current_sub_question_index": 0,
            "solved_sub_questions": [],
            "key_entities": {},
            "failed_attempts": [],
            "recent_interaction_history": []
        }

    def _update_scratchpad(self, **kwargs):
        self.scratchpad.update(kwargs)

    def _perform_initial_planning(self, user_query: str) -> List[str]:
        print("\n===== Performing Initial Planning... =====")
        planning_prompt = f"""
        # 指令: 问题分解
        用户提出了以下复杂问题: "{user_query}"
        你的任务是, 将这个问题分解成一个有序的、逻辑上层层递进的子问题列表。每个子问题的答案都应该是下一个子问题的先决条件。请将你的计划以编号列表的形式输出。确保计划的最后一步是直接回答用户的原始问题。
        """
        messages = [{"role": "user", "content": planning_prompt}]
        response = self.llm_call(messages, stream=True)
        response.raise_for_status()
        plan_text = _get_streamed_content(response)
        plan = [line.strip() for line in plan_text.splitlines() if re.match(r'^\s*\d+\.', line)]
        print(f"Generated Plan:\n{json.dumps(plan, indent=2, ensure_ascii=False)}")
        return plan

    def _build_contextual_prompt(self) -> List[Dict[str, str]]:
        current_plan_step = ""
        if self.scratchpad['reasoning_plan'] and 0 <= self.scratchpad['current_sub_question_index'] < len(self.scratchpad['reasoning_plan']):
            current_plan_step = self.scratchpad['reasoning_plan'][self.scratchpad['current_sub_question_index']]

        # ADDED: Stricter instruction in the prompt to prevent premature finishing.
        memory_summary = f"""
任务状态摘要
总体目标: {self.scratchpad['overall_objective']}

解题计划: {json.dumps(self.scratchpad['reasoning_plan'], ensure_ascii=False)}

当前进展: 已完成计划中的第 {self.scratchpad['current_sub_question_index']} 步。

已知事实 (关键实体): {json.dumps(self.scratchpad['key_entities'], ensure_ascii=False, indent=2)}

当前任务: 正在尝试解决第 {self.scratchpad['current_sub_question_index'] + 1} 步: "{current_plan_step}"

**关键指令: 你的唯一任务是严格按照计划执行下一步。在所有计划步骤全部完成之前，绝对不允许调用 `final_answer` 工具！**
"""
        messages = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {"role": "user", "content": memory_summary}
        ]
        messages.extend(self.scratchpad['recent_interaction_history'])
        return messages

    def _perform_reflection(self, failed_action: Dict, error_message: str) -> str:
        print("\n===== Performing Reflection... =====")
        reflection_prompt = f"""
指令: 反思与修正
上一步的行动失败了。

失败的行动:
{json.dumps(failed_action, indent=2, ensure_ascii=False)}
导致的结果:
Observation: {error_message}

现在, 请进行深入反思:
根本原因分析: 批判性地分析这次失败的根本原因是什么？
提出修正方案: 基于你的分析，提出一个全新的、经过修正的Action来克服这个失败。
现在, 请输出你修正后的Thought和Action。
"""
        context = self._build_contextual_prompt()
        context.append({"role": "user", "content": reflection_prompt})
        response = self.llm_call(context, stream=True)
        response.raise_for_status()
        return _get_streamed_content(response)

    def _parse_llm_response(self, response_text: str) -> (str, Dict[str, Any]):
        thought_match = re.search(r"Thought:\s*(.*?)\s*Action:", response_text, re.DOTALL)
        action_match = re.search(r"Action:\s*```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if not thought_match or not action_match:
            raise ValueError("Could not parse Thought or Action from LLM response. The format is incorrect.")
        thought = thought_match.group(1).strip()
        action_str = action_match.group(1).strip()
        try:
            action = json.loads(action_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in Action block: {e}\nContent: {action_str}")
        return thought, action

    def run(self, user_query: str):
        self.__init__(self.max_turns)
        plan = self._perform_initial_planning(user_query)
        self._update_scratchpad(overall_objective=user_query, reasoning_plan=plan)

        turn_count = 0
        while turn_count < self.max_turns:
            turn_count += 1
            print(f"\n{'='*20} Turn {turn_count} {'='*20}")

            if not self.scratchpad['reasoning_plan']:
                print("Error: The initial planning step failed to produce a plan.")
                return {"answer": "I could not create a plan to answer this question.", "sources": []}

            context = self._build_contextual_prompt()

            try:
                response = self.llm_call(context, stream=True)
                response.raise_for_status()
                llm_text = _get_streamed_content(response)
                print(f"LLM Raw Response:\n{llm_text}")
                thought, action = self._parse_llm_response(llm_text)
                print(f"Thought: {thought}")
                print(f"Action: {json.dumps(action, indent=2, ensure_ascii=False)}")
            except Exception as e:
                print(f"Error in LLM call or parsing: {e}. Moving to next turn.")
                self.scratchpad['recent_interaction_history'].append({"role": "user", "content": f"Observation: Encountered an error - {e}. You must recover from this."})
                continue

            self.scratchpad['recent_interaction_history'].append({"role": "assistant", "content": llm_text})

            tool_name = action.get('tool_name')
            params = action.get('parameters', {})

            if tool_name == 'final_answer':
                current_index = self.scratchpad['current_sub_question_index']
                plan_length = len(self.scratchpad['reasoning_plan'])
                
                if current_index >= plan_length - 3:
                    print("\n===== Final Answer Received =====")
                    return action['parameters']
                else:
                    print("\n[GUARDRAIL] Agent attempted to call final_answer prematurely. Rejecting action.")
                    error_message = (f"拒绝执行: 你在计划的第 {current_index + 1} 步就尝试调用 'final_answer'，"
                                     f"但整个计划有 {plan_length} 步。你必须严格完成所有步骤才能给出最终答案。"
                                     f"请继续执行下一步计划：'{self.scratchpad['reasoning_plan'][current_index]}'，如果你已经确定解决了最终问题，那么请你对答案与初始问题进行验证。")
                    self.scratchpad['recent_interaction_history'].append({"role": "user", "content": f"Observation: {error_message}"})
                    continue 

            if tool_name in self.tools:
                try:
                    if tool_name == 'jina_reader':
                        params['question'] = self.scratchpad['reasoning_plan'][self.scratchpad['current_sub_question_index']]
                    
                    result = self.tools[tool_name](**params)
                    print(f"Observation: {result}")

                    if not result or (isinstance(result, str) and result.startswith('{"error"')):
                        raise ValueError(f"Tool execution failed or returned an error: {result}")
                    
                    self.scratchpad['recent_interaction_history'].append({"role": "user", "content": f"Observation:\n{result}"})

                    current_index = self.scratchpad['current_sub_question_index']
                    if current_index < len(self.scratchpad['reasoning_plan']) - 1:
                        self.scratchpad['current_sub_question_index'] += 1
                        print(f"\n[Progress] Moved to sub-question #{self.scratchpad['current_sub_question_index'] + 1}")
                    else:
                        print("\n[Progress] All planned steps are complete. The agent should now provide the final_answer.")

                except Exception as e:
                    print(f"Error executing tool '{tool_name}': {e}")
                    self.scratchpad['failed_attempts'].append({'action': action, 'error': str(e)})
                    self.scratchpad['recent_interaction_history'].append({"role": "user", "content": f"Observation: The last action failed with error: {e}. You must reflect and try a different approach."})
                    continue
            else:
                print(f"Unknown tool: {tool_name}")
                self.scratchpad['recent_interaction_history'].append({"role": "user", "content": f"Observation: You tried to use an unknown tool named '{tool_name}'. Available tools are: {list(self.tools.keys())}"})

        return "Agent stopped after reaching max turns."
