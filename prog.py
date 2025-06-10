"""Example of prompting a model"""
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

async def main():
    """Example main function"""

    parser = argparse.ArgumentParser(
        description="takes an optional prompt argument")
    parser.add_argument("prompt", type=str, help="the prompt to use", default="", nargs="?")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt != "" else "roll a die"

    context = support.AIContext(Model)

    async with context.session:
        response = await support.single_shot_ask(context, prompt)

    print(response)

if __name__ == "__main__":
    asyncio.run(main())
