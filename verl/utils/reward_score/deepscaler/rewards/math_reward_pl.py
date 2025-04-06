"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from verl.utils.reward_score.deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from verl.utils.reward_score.deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from verl.utils.reward_score.deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from verl.utils.reward_score.deepscaler.system_prompts import ORM_PROMPT
# from verl.utils.reward_score.deepscaler.utils import call_gemini_llm, call_oai_rm_llm

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        print("*"*80, "\nproblem: ", problem, '\n', "*"*80, "\nmodel_response: ", model_response, "*"*80)
        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            preference_level = 4
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False), preference_level
        
        model_answer = extract_answer(model_solution)
        print("*"*80, "\n", "model_answer", model_answer, '\n', "*"*80, "\n")
        if model_answer == "":
            print("*"*80, "\n", f"model_answer: {model_answer} is empty\n", "*"*80, "\n", )
            preference_level = 3
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False), preference_level
        if model_answer is None:
            preference_level = 4
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False), preference_level

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            preference_level = 3
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False), preference_level
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            preference_level = 3
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False), preference_level
        print("*"*80, "\n", "processed_ground_truths", processed_ground_truths, '\n', "*"*80, "\n")
        
        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                preference_level = 1
                return RewardOutput(reward=self.config.correct_reward, is_correct=True), preference_level
            else:
                preference_level = 2
                return RewardOutput(reward=self.config.incorrect_reward, is_correct=False), preference_level

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            assert False, "use_math_orm is not supported for rule based RL. "
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        preference_level = 1
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True), preference_level
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        preference_level = 1
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True), preference_level
                    continue
        
        preference_level = 3  
        print("preference level", preference_level, '\n', "*"*80, "\n")
        return RewardOutput(reward=self.config.unk_error_reward, is_correct=False), preference_level

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False, preference_reward=False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    print("*"*80, "\n", "ground_truth: ", ground_truth, '\n', "*"*80, "\n")
    reward_response, preference_level = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    if not preference_reward:
        return reward_response.reward
    else:
        return reward_response.reward, preference_level

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer analysis and answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively and the final answer is enclosed within \boxed{}, i.e., <think> reasoning process here </think><answer> answer analysis here. The final answer is \boxed{answer here} </answer>.  Now the user asks you to solve a math problem. Please think step by step. \n<|im_end|>\n<|im_start|>user\nDoug constructs a square window using $8$ equal-size panes of glass. The ratio of the height to width for each pane is $5 : 2$, and the borders around and between the panes are $2$ inches wide. In inches, what is the side length of the square window? Let's think step by step and output the final answer within \boxed{}.\n<|im_end|>\n<|im_start|>assistant\n<think>\\boxed{}.\\boxed{}.\\boxed{}. Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
