"""The tasks A–E and the learn/unlearn streams (PLAN.md, week 1 and §8)."""

# All ten LIBERO-Goal tasks, by LIBERO task name. The instruction is the name with spaces.
GOAL_TASK_NAMES = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
]

# Letter -> LIBERO-Goal task. PROVISIONAL (PLAN.md Day 1: "fix the A–E task mapping").
# Changing this after runs exist makes their results meaningless, so treat it as frozen once set.
TASKS = {
    "A": "open_the_middle_drawer_of_the_cabinet",
    "B": "put_the_bowl_on_the_stove",
    "C": "put_the_wine_bottle_on_top_of_the_cabinet",
    "D": "open_the_top_drawer_and_put_the_bowl_inside",
    "E": "put_the_bowl_on_top_of_the_cabinet",
}


def instruction(letter: str) -> str:
    return TASKS[letter].replace("_", " ")


# Streams of ops (see vlu/nodes.py for the op syntax). U ops get their method appended when a
# stream is expanded for one unlearning method, e.g. "UB" -> "UB-ga".
STREAMS = {
    "S1": ["LA", "LB", "LC", "UB", "LD", "LE", "UE"],
}
