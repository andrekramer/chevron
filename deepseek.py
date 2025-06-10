"""Deepseek Moedels"""
import support

deepseek_api_key = support.read_file_as_string("deepseek-api-key").strip()

URL = "https://api.deepseek.com/chat/completions"

class Deepseek(support.Model):
    """Deepseek AI Model"""
    name = "deepseek"
    model = "deepseek-chat"

    text_field = "content"

    @classmethod
    def make_query(cls, text):
        """make a query for Deepseek"""
        return support.make_openai_std_query(text, cls.model)

    @staticmethod
    async def ask(session, query):
        """make a request to Deepseek using http post"""
        headers = {
          "Content-Type": "application/json",
          "Authorization": "Bearer " + deepseek_api_key
        }
        # print(f"deepseek ask: {query}")
        return await support.ask(URL, session, query, headers)

class Deepseek2(Deepseek):
    """Deepseek Reasoner"""
    name = "deepseek2"
    model = "deepseek-reasoner"
    reasoning_text_field = "reasoning_content"
