"""Example of teacher / student prompting a model. Hegel may have said Master / Slave dialectic."""
import argparse
import asyncio

import support

from localhost import LocalHost
from openai import Openai4
from gemini import Gemini3
from claud import Claud
from llama import Llama2
from deepseek import Deepseek2
from grok import Grok2
from hugface import HugFace

Model = Gemini3
Teacher_Model = Model
Student_Model = Model


TEACHER_PROMPT = "You are a teacher, consider the self and the world with a student as your dialectic opponent."
STUDENT_PROMPT = "You are a student, consider the self and the world with a teacher as your dialectic opponent."
BOTH_PROMPT = "You and your opponent are internally an actor and a critic. So a split or double dyad.\n" + \
    "Hegel's master / slave and theory / praxis dynamics in a repeated thesis / antithesis / synthesis dialectic.\n" + \
    "The Teacher is the world-model and the Student is the self-model.\n" + \
    "Each actor (praxis) and critic (theory) considers itself and the others in the following relations.\n" + \
    "Where T is the teacher and S is the student.\n" + \
    "A is Actor and C is critic. \"->\" means considers:"
TEACHER_CONTEXT = \
"TA->TA\n" + \
"TA->SA\n" + \
"TA->SC\n" + \
"TA->TC\n" + \
"TC->TC\n" + \
"TC->TA\n" + \
"TC->SA\n" + \
"TC->SC\n"
STUDENT_CONTEXT = \
"SA->SA\n" + \
"SA->SC\n" + \
"SA->MC\n" + \
"SA->MA\n" + \
"SC->SC\n" + \
"SC->MA\n" + \
"SC->MC\n" + \
"SC->SA\n"
DIALECTIC = "The current state of the dialectic is:"

TEACHER = TEACHER_PROMPT + "\n" + BOTH_PROMPT + "\n" + TEACHER_CONTEXT + "\n" + DIALECTIC + "\n"
STUDENT = STUDENT_PROMPT + "\n" + BOTH_PROMPT + "\n" + STUDENT_CONTEXT + "\n" + DIALECTIC + "\n"

TURN = "\n" + "---------" + "\n"
TEACHER_TURN = " teacher -> student"
STUDENT_TURN = " student -> teacher"

turn_count = 1

async def main():
    """The main function for the teacher-student dialectic example."""
    global turn_count

    teacher = True
    parser = argparse.ArgumentParser(description="takes an optional initial prompt argument")
    parser.add_argument("prompt", type=str, help="the initial prompt to use", default="", nargs="?")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt != "" else "free will is an illusion"

    print("initial prompt > " + prompt)

    while True:
        if teacher:
          dialectic_prompt = TEACHER + "\n" + prompt
        else:
          dialectic_prompt = STUDENT + "\n" + prompt

        print("\n" + ("teacher" if teacher else "student") + "prompt > " + dialectic_prompt)

        next = input("hit enter to continue, or type . to exit ")
        if next == ".":
            break

        context = support.AIContext(Teacher_Model if teacher else Student_Model)

        async with context.session:
            response = await support.single_shot_ask(context, dialectic_prompt)

        print(response)
        teacher = not teacher

        turn = TURN + "\n"
        if teacher:
          turn += str(turn_count) + TEACHER_TURN + TURN
        else:
          turn += str(turn_count) + STUDENT_TURN + TURN
        prompt = prompt + turn + response
        turn_count += 1

if __name__ == "__main__":
  asyncio.run(main())
