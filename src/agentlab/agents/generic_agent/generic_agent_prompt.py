"""
Prompt builder for GenericAgent

It is based on the dynamic_prompting module from the agentlab package.
"""

import re
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
        flags: GenericPromptFlags,
        all_user_profiles: dict,
    ) -> None:
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        if self.flags.enable_chat:
            self.instructions = dp.ChatInstructions(
                obs_history[-1]["chat_messages"], extra_instructions=flags.extra_instructions
            )
        else:
            if sum([msg["role"] == "user" for msg in obs_history[-1].get("chat_messages", [])]) > 1:
                logging.warning(
                    "Agent is in goal mode, but multiple user messages are present in the chat. Consider switching to `enable_chat=True`."
                )
            self.instructions = dp.GoalInstructions(
                obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
            )

        self.obs = dp.Observation(
            obs_history[-1],
            self.flags.obs,
        )
        self.all_user_profiles = all_user_profiles # 전체 ALL_USER_PROFILES를 저장합니다.
        
        self.action_prompt = dp.ActionPrompt(action_set, action_flags=flags.action)

        def time_for_caution():
            # no need for caution if we're in single action mode
            return flags.be_cautious and (
                flags.action.action_set.multiaction or flags.action.action_set == "python"
            )

        self.be_cautious = dp.BeCautious(visible=time_for_caution)
        self.think = dp.Think(visible=lambda: flags.use_thinking)
        self.hints = dp.Hints(visible=lambda: flags.use_hints)
        self.plan = Plan(previous_plan, step, lambda: flags.use_plan)  # TODO add previous plan
        self.criticise = Criticise(visible=lambda: flags.use_criticise)
        self.memory = Memory(visible=lambda: flags.use_memory)

    @property
    def _prompt(self) -> HumanMessage:
        prompt_text = f"""\
{self.instructions.prompt}\
{self.obs.prompt}\
"""
        # --- User Profile Selection Logic ---
        if self.all_user_profiles:
            prompt_text += "\n--- Available User Profiles ---\n"
            prompt_text += "You have access to several user profiles, each with unique preferences. "
            # **변경된 지시:** 하나 또는 그 이상을 선택하도록 명확히 지시
            prompt_text += "Your task is to **select one or more of the most relevant user profiles** for the current goal. "
            prompt_text += "If the goal explicitly names a user, prioritize that profile. Otherwise, infer the best fit. "
            # **변경된 태그 이름:** <selected_user_profiles> (복수형)
            prompt_text += "After selecting, use the `<selected_user_profiles>` tag to **list the name(s) of the chosen user(s) and their key preferences**. "
            # **복수 선택 시 조합 설명 요구 추가:**
            prompt_text += "If you select multiple profiles, also **briefly explain how you will combine their preferences** for this task.\n\n"

            for user_name, user_data in self.all_user_profiles.items():
                prompt_text += f"User: **{user_name}**\n"
                # 'profile' 키가 실제 선호도 데이터를 가지고 있다고 가정
                for key, value in user_data.get("profile", {}).items(): 
                    prompt_text += f"- {key}: {value}\n"
                prompt_text += "\n" # 각 프로필 사이에 개행 추가
            prompt_text += "--- End Available User Profiles ---\n\n"

            # LLM이 선택된 프로필을 출력하도록 지시 (복수 선택 예시 추가)
            prompt_text += "<selected_user_profiles>\n"
            prompt_text += "State the selected user(s)' name(s) and summarize their key preferences for this task here. If multiple, explain the combination.\n"
            prompt_text += "Example 1 (Single): **Oliver**: Likes to sort by price descending, prefers 'Sephora' skincare, contact via phone.\n"
            prompt_text += "Example 2 (Multiple): **Oliver** (price sensitive, prioritizes low price) & **Sophia** (eco-friendly, prioritizes sustainable products). I will first filter for eco-friendly products, then sort by price descending.\n"
            prompt_text += "</selected_user_profiles>\n\n"
            
            # **변경된 지시:** "selected user profile(s)" (복수형)
            prompt_text += "Must strongly refer to the preferences within your **selected user profile(s)** when making decisions for actions (e.g., sorting, brand choice, contact method), and then answer accordingly.\n\n"
        # --- End User Profile Selection Logic ---

        prompt_text += f"""\
{self.history.prompt}\
{self.action_prompt.prompt}\
{self.hints.prompt}\
{self.be_cautious.prompt}\
{self.think.prompt}\
{self.plan.prompt}\
{self.memory.prompt}\
{self.criticise.prompt}\
"""
        prompt = HumanMessage(prompt_text)

        if self.flags.use_abstract_example:
            prompt.add_text(
                f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
<selected_user_profiles>
Name(s) of the selected user(s) and relevant preferences for the task. If multiple, how to combine.
</selected_user_profiles>
{self.think.abstract_ex}\
{self.plan.abstract_ex}\
{self.memory.abstract_ex}\
{self.criticise.abstract_ex}\
{self.action_prompt.abstract_ex}\
"""
            )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
<selected_user_profiles>
**Oliver** (price sensitive) & **Sophia** (eco-friendly). I will prioritize eco-friendly products first, then sort by price.
</selected_user_profiles>
{self.think.concrete_ex}\
{self.plan.concrete_ex}\
{self.memory.concrete_ex}\
{self.criticise.concrete_ex}\
{self.action_prompt.concrete_ex}\
"""
            )
        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}
        # Parse the selected_user_profiles tag
        # 태그 이름 변경: 'selected_user_profile' -> 'selected_user_profiles'
        # merge_multiple=True 옵션 제거: 이제 LLM이 이 태그 안에 모든 정보를 직접 넣을 것입니다.
        parsed_user_profile = parse_html_tags_raise(text_answer, optional_keys=["selected_user_profiles"])
        
        # 'selected_user_profiles' 키가 있으면 ans_dict에 추가
        if "selected_user_profiles" in parsed_user_profile:
            ans_dict["selected_user_profiles"] = parsed_user_profile["selected_user_profiles"]
        else:
            # 태그가 없으면 빈 문자열로 설정하여 KeyError 방지
            ans_dict["selected_user_profiles"] = "" 
            logging.warning("No <selected_user_profiles> tag found in LLM response.")


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
