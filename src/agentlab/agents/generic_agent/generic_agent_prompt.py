"""
Prompt builder for GenericAgent

It is based on the dynamic_prompting module from the agentlab package.
"""
import os
import base64
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import logging
from dataclasses import dataclass

from browsergym.core import action
from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.llm_utils import HumanMessage, parse_html_tags_raise


@dataclass
class GenericPromptFlags(dp.Flags):
    """
    A class to represent various flags used to control features in an application.

    Attributes:
        use_plan (bool): Ask the LLM to provide a plan.
        use_criticise (bool): Ask the LLM to first draft and criticise the action before producing it.
        use_thinking (bool): Enable a chain of thoughts.
        use_concrete_example (bool): Use a concrete example of the answer in the prompt for a generic task.
        use_abstract_example (bool): Use an abstract example of the answer in the prompt.
        use_hints (bool): Add some human-engineered hints to the prompt.
        enable_chat (bool): Enable chat mode, where the agent can interact with the user.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.
        be_cautious (bool): Instruct the agent to be cautious about its actions.
        extra_instructions (Optional[str]): Extra instructions to provide to the agent.
        add_missparsed_messages (bool): When retrying, add the missparsed messages to the prompt.
        flag_group (Optional[str]): Group of flags used.
    """

    obs: dp.ObsFlags
    action: dp.ActionFlags
    use_plan: bool = False  #
    use_criticise: bool = False  #
    use_thinking: bool = False
    use_memory: bool = False  #
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    enable_chat: bool = False
    max_prompt_tokens: int = None
    be_cautious: bool = True
    extra_instructions: str | None = None
    add_missparsed_messages: bool = True
    max_trunc_itr: int = 20
    flag_group: str = None


# ===== 전역 설정: 이미지 임베딩 및 FAISS 인덱스 =====
IMG_DIR = "/home/mila/s/shind/scratch/RAG_practice/image/user1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 512  # CLIP ViT-B/32 기준

# CLIP 모델 로드
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# 이미지 파일 로드
IMAGE_PATHS = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".png")]

# 이미지 임베딩 계산 (한 번만 수행)
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embedding = CLIP_MODEL.get_image_features(**inputs)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy()

EMBEDDINGS = np.vstack([embed_image(p) for p in IMAGE_PATHS]).astype("float32")

# FAISS 인덱스 생성 (한 번만 수행)
INDEX = faiss.IndexFlatIP(EMBEDDING_DIM)
if DEVICE == "cuda":
    res = faiss.StandardGpuResources()
    INDEX = faiss.index_cpu_to_gpu(res, 0, INDEX)
INDEX.add(EMBEDDINGS)
print(f"FAISS index created for {INDEX.ntotal} images.")

# ===== RAG 검색 + 요약 함수 =====
def retrieve_and_summarize(task_instruction: str) -> str:
    query = f"Which image is the most relevant to following task instruction?: '{task_instruction}'"
    text_inputs = CLIP_PROCESSOR(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_embedding = CLIP_MODEL.get_text_features(**text_inputs)
    text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy()

    # 상위 3개 이미지 검색
    D, I = INDEX.search(text_embedding, k=3)
    best_image_paths = [IMAGE_PATHS[idx] for idx in I[0]]

    # 이미지를 base64로 변환
    image_contents = []
    for path in best_image_paths:
        with open(path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_contents.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        )

    # OpenAI API 호출
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Look at the following images and describe my preferences in one sentence. "
                    "Speak as if you are me (use first-person narration, starting with 'I ...')."
                    "Always start your answer with 'I' and write in first-person narration."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "From the following images, describe my preferences in one sentence. "
                            "Write as if I am explaining them myself (use first-person narration)."
                        ),
                    },
                    *image_contents,
                ],
            },
        ],
    )
    return response.choices[0].message.content


# ===== ImageSummary PromptElement =====
class ImageSummary(dp.PromptElement):
    def __init__(self, summary: str, visible: bool = True):
        super().__init__(visible=visible)
        self._prompt = f"<image_summary>\n{summary}\n</image_summary>"

    def _parse_answer(self, text_answer):
        return {}


# ===== 최종 MainPrompt 클래스 =====
class MainPrompt(dp.Shrinkable):
    def __init__(
        self,
        action_set: AbstractActionSet,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        previous_plan: str,
        step: int,
        flags: dp.Flags,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)

        # Chat / Goal instruction 분기
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning("Agent is in goal mode, but multiple user messages exist. Consider enable_chat=True.")
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(obs_history[-1], self.flags.obs)
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        def time_for_caution():
            return flags.be_cautious and (flags.action.action_set.multiaction or flags.action.action_set == "python")

        self.be_cautious = dp.BeCautious(visible=time_for_caution)
        self.think = dp.Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        self.plan = Plan(previous_plan, step, lambda: flags.use_plan)
        self.criticise = Criticise(visible=lambda: flags.use_criticise)
        self.memory = Memory(visible=lambda: flags.use_memory)

        # === task instruction 기반 image summary 생성 ===
        task_instruction = obs_history[-1]["goal_object"]
        image_summary_text = retrieve_and_summarize(task_instruction)
        self.image_summary = ImageSummary(image_summary_text, visible=True)

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"{self.image_summary.prompt}"
            f"{self.obs.prompt}{self.history.prompt}{self.action_prompt.prompt}{self.hints.prompt}"
            f"{self.be_cautious.prompt}{self.think.prompt}{self.plan.prompt}{self.memory.prompt}"
            f"{self.criticise.prompt}"
        )
        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.think.parse_answer(text_answer))
        ans_dict.update(self.plan.parse_answer(text_answer))
        ans_dict.update(self.memory.parse_answer(text_answer))
        ans_dict.update(self.criticise.parse_answer(text_answer))
        ans_dict.update(self.action_prompt.parse_answer(text_answer))
        return ans_dict



class Memory(dp.PromptElement):
    _prompt = ""  # provided in the abstract and concrete examples

    _abstract_ex = """
<memory>
Write down anything you need to remember for next steps. You will be presented
with the list of previous memories and past actions. Some tasks require to
remember hints from previous steps in order to solve it.
</memory>
"""

    _concrete_ex = """
<memory>
I clicked on bid "32" to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Plan(dp.PromptElement):
    def __init__(self, previous_plan, plan_step, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self.previous_plan = previous_plan
        self._prompt = f"""
# Plan:

You just executed step {plan_step} of the previously proposed plan:\n{previous_plan}\n
After reviewing the effect of your previous actions, verify if your plan is still
relevant and update it if necessary.
"""

    _abstract_ex = """
<plan>
Provide a multi step plan that will guide you to accomplish the goal. There
should always be steps to verify if the previous action had an effect. The plan
can be revisited at each steps. Specifically, if there was something unexpected.
The plan should be cautious and favor exploring befor submitting.
</plan>

<step>Integer specifying the step of current action
</step>
"""

    _concrete_ex = """
<plan>
1. fill form (failed)
    * type first name
    * type last name
2. Try to activate the form
    * click on tab 2
3. fill form again
    * type first name
    * type last name
4. verify and submit
    * verify form is filled
    * submit if filled, if not, replan
</plan>

<step>2</step>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["plan", "step"])


class Criticise(dp.PromptElement):
    _prompt = ""

    _abstract_ex = """
<action_draft>
Write a first version of what you think is the right action.
</action_draft>

<criticise>
Criticise action_draft. What could be wrong with it? Enumerate reasons why it
could fail. Did your past actions had the expected effect? Make sure you're not
repeating the same mistakes.
</criticise>
"""

    _concrete_ex = """
<action_draft>
click("32")
</action_draft>

<criticise>
click("32") might not work because the element is not visible yet. I need to
explore the page to find a way to activate the form.
</criticise>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["action_draft", "criticise"])
