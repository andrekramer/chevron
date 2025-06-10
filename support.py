"""support for model comparisons"""

from os import getenv, path
import json
import aiohttp

DEBUG = False

# Set to place where repo copy is located when running as MCP server.
# Can be set to None when running as a script in same dir.
# Or if set as an env var (called FILENAME).
FILEPATH = None

filepath = getenv("FILEPATH", FILEPATH)

def escape(s):
    """remove line breaks from string and escape \""""
    str1 = s.replace("\n", "\\n")
    str2 = str1.replace('"', '\\"')
    return str2

def unescape(s):
    """remove line break and double quote escaping \""""
    str1 = s.replace("\\n", "\n")
    str2 = str1.replace('\\"', '"')
    return str2

class Model:
    """Base class for all AI models"""

    @staticmethod
    def make_query(text):
        """abstract make query method"""
        raise RuntimeError("Not implemented")

    @staticmethod
    async def ask(session, query):
        """abstract ask method"""
        raise RuntimeError("Not implemented")

    # fields to implement: name, model, text_field

TIMEOUT_IN_SECONDS = 180
timeout = aiohttp.ClientTimeout(total=TIMEOUT_IN_SECONDS)

def get_session():
    """get http async session with configured timeout"""
    return aiohttp.ClientSession(timeout=timeout)

class AIContext:
    """AI Context"""
    def __init__(self, model):
        self.model = model
        self.session = get_session()

    def get_model(self):
        """get the model"""
        return self.model

    def set_model(self, model):
        """set the model"""
        self.model = model

def serialize(json_object):
    """serialize an object to json"""
    return json.dumps(json_object)

def make_openai_std_query_with_str_concat(text, model):
    """openai compatible queries"""
    return  "{ \"model\": \"" + model + \
             "\", \"messages\": [{\"role\": \"user\", \"content\": \"" + \
            text + "\"}]}"

def make_openai_std_query_from_text(text, model):
    """create an openai compatible query"""
    obj = { "model": model }
    return make_std_query_from_object(obj, text)

def make_std_query_from_object(obj, text):
    """make a chat query"""
    message = { "role": "user" }
    message["content"] = text
    messages = []
    messages.append(message)
    obj["messages"] = messages

    return serialize(obj)

make_openai_std_query = make_openai_std_query_from_text

def read_file_as_string(filename):
    """read a file as string"""

    filename = path.join(filepath, filename) if filepath is not None else filename
    try:
        with open(filename, mode='r', encoding="utf-8") as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{filename}': {e}")
        return None

def write_file_as_string(filename, content):
    """write a file as string"""
    try:
        with open(filename, mode='w', encoding="utf-8") as file:
            file.write(content)
    except Exception as e:
        print(f"An error occurred while writing '{filename}': {e}")
        raise e

async def ask(url, session, query, headers):
    """ask via async http post"""
    try:
        async with session.post(url, data=query.encode(), headers=headers) as response:
            # print(f"Fetch {url}: Status code {response.status}")
            if response.status != 200:
                print(f"Error while fetching {url}: {response.status}")
                return "{\"error\": " + str(response.status) + "}"
            return await response.text()
    except Exception as e:
        print(f"An error occurred while fetching {url}: {e.__class__.__name__}")
        return "{ \"error\": \"" + e.__class__.__name__ + "\" }"

async def single_shot_ask(context, prompt, allow_not_found=True):
    """ask one model a single question"""
    response = await context.model.ask(context.session, context.model.make_query(prompt))

    json_data = json.loads(response) if response is not None and response != "" else {}
    if DEBUG:
        json_formatted_str = json.dumps(json_data, indent=2)
        print(json_formatted_str)

    text = search_json(json_data, context.model.text_field)
    if not allow_not_found and (text is None or text.strip() == ""):
        raise Exception("No response text from model")
    return text

def search_json(json_data, target_key):
    """Recursively searches a JSON object for a key."""
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == target_key:
                return value
            if isinstance(value, (dict, list)):  # Recurse into nested structures
                result = search_json(value, target_key)
                if result is not None:
                    return result
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, (dict, list)):
                result = search_json(item, target_key)
                if result is not None:
                    return result
    return None  # Key not found

def extract_tag(text, tag):
    """extract text between tags"""
    start_tag = "<" + tag + ">"
    end_tag = "</" + tag + ">"
    start = text.find(start_tag)
    if start == -1:
        return None
    start += len(start_tag)
    end = text.find(end_tag)
    if end == -1:
        return None
    return text[start:end]
