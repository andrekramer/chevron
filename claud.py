"""Claud AI Model"""
import support

claud_api_key = support.read_file_as_string("claud-api-key").strip()

URL = "https://api.anthropic.com/v1/messages"

class Claud(support.Model):
    """Claud AI Model"""
    name = "claud"
    model = "claude-3-5-sonnet-20241022"
    text_field = "text"

    @classmethod
    def make_query(cls, text):
        """make claud query"""
        obj = { "model": cls.model, "max_tokens": 2048 }
        return support.make_std_query_from_object(obj, text)

    @staticmethod
    async def ask(session, query):
        """perform query"""
        headers = {
          "Content-Type": "application/json",
          "x-api-key": claud_api_key,
          "anthropic-version": "2023-06-01"
        }
        return await support.ask(URL, session, query, headers)

class Claud2(Claud):
    """Claud AI 3.7 Model"""
    name = "claud2"
    model = "claude-3-7-sonnet-20250219"

class Claud3(Claud):
    """Claud AI Model 3.7 Thinking"""
    name = "claud3"
    # model = "claude-3-7-sonnet-20250219" same as Claud2
    reasoning_text_field = "thinking"

    @classmethod
    def make_query(cls, text):
        """make claud query with thinking"""
        obj = {
            "model": cls.model, 
            "max_tokens": 20000,
            "thinking": {
                 "type": "enabled",
                 "budget_tokens": 10000
            }
        }
        return support.make_std_query_from_object(obj, text)
