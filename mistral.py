"""Mistral models"""
import support

mistral_api_key = support.read_file_as_string("mistral-api-key").strip()

URL = "https://api.mistral.ai/v1/chat/completions"

class Mistral(support.Model):
    """mistral large latest"""
    name = "mistral"
    model = "mistral-large-latest"
    text_field = "content"

    @classmethod
    def make_query(cls, text):
        """make a query for Mistral"""
        return support.make_openai_std_query(text, cls.model)

    @staticmethod
    async def ask(session, query):
        """make a request to Mistral using http post"""
        headers = {
          "Content-Type": "application/json",
          "Accept": "application/json",
          "Authorization": "Bearer " + mistral_api_key
        }
        return await support.ask(URL, session, query, headers)
