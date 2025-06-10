"""Llama models from Meta"""
import support

llama_api_key = support.read_file_as_string("llama-api-key").strip()

URL = "https://api.llama-api.com/chat/completions"

class Llama(support.Model):
    """Llama 3.2b"""
    name = "llama"
    model = "llama3.2-3b"

    text_field = "content"

    @classmethod
    def make_query(cls, text):
        """make a query for LLama"""
        return support.make_openai_std_query(text, cls.model)

    @staticmethod
    async def ask(session, query):
        """make a request to Llama using http post"""
        headers = {
          "Content-Type": "application/json",
          "Authorization": "Bearer " + llama_api_key
        }
        return await support.ask(URL, session, query, headers)

class Llama2(Llama):
    """llama3.3-70b"""
    name = "llama2"
    model = "llama3.3-70b"
