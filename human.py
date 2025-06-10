"""Human plug-in as a model"""
import support

SEPARATOR = "-" * 80

class Human(support.Model):
    """Human as a model"""
    name = "human"
    model = "human-1.0-latest"
    text_field = "text"

    @staticmethod
    def make_query(text):
        """make a query"""
        return text

    @classmethod
    async def ask(cls, session, query):
        """make a model request by getting input from a command line"""
        print(SEPARATOR)
        print(support.unescape(query))
        response = input("Human: ")
        return '{ "text": "' + support.escape(response) + '" }'
