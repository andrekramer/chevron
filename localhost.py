"""Some Local Models"""
import support

URL = "http://127.0.0.1:1234/v1/chat/completions"

class LocalHost(support.Model):
    """Local AI Model"""
    name = "localhost"
    model = "localmodel-chat"

    text_field = "content"

    @staticmethod
    def make_query(text):
        """make a query for local model"""
        obj = {}
        return support.make_std_query_from_object(obj, text)

    @staticmethod
    async def ask(session, query):
        """make a request to LocalHost using http post"""
        headers = {
          "Content-Type": "application/json"
        }
        return await support.ask(URL, session, query, headers)
