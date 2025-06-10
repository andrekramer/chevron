"""Grok models from X"""
import support

grok_api_key = support.read_file_as_string("grok-api-key").strip()

URL = "https://api.x.ai/v1/chat/completions"

class Grok(support.Model):
    """grok beta"""
    name = "grok"
    model = "grok-beta"
    text_field = "content"

    @classmethod
    def make_query(cls, text):
        """make a query for Grok"""
        return support.make_openai_std_query(text, cls.model)

    @staticmethod
    async def ask(session, query):
        """make a request to Grok using http post"""
        headers = {
          "Content-Type": "application/json",
          "Authorization": "Bearer " + grok_api_key
        }
        return await support.ask(URL, session, query, headers)

class Grok2(Grok):
    """Grok 2 latest"""
    name = "grok2"
    model = "grok-2-latest"
