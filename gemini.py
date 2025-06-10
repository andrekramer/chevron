"""Gemini models from Meta"""
import support

gemini_api_key = support.read_file_as_string("gemini-api-key").strip()

class Gemini(support.Model):
    """Meta Geminin 1.5 flast latest"""
    name = "gemini"
    model = "gemini-1.5-flash-latest"
    text_field = "text"

    @staticmethod
    def make_query(text):
        """make a query"""
        obj = {}
        part = { "text": text }
        contents = []
        parts = []
        parts.append(part)
        contents.append({ "parts": parts })
        obj["contents"] = contents
        return support.serialize(obj)

    @classmethod
    async def ask(cls, session, query):
        """make a request using http post"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/" + \
              cls.model + ":generateContent?key=" + gemini_api_key
        print("query url " + url)
        headers = {
            "Content-Type": "application/json"
        }
        return await support.ask(url, session, query, headers)

class Gemini2(Gemini):
    """Meta Gemini 2.0 flash experimental"""
    name = "gemini2"
    model = "gemini-2.0-flash"

class Gemini3(Gemini):
    """Meta Gemini 2.5 flash"""
    name = "gemini3"
    model = "gemini-2.5-flash-preview-05-20"

class Gemini4(Gemini):
    """Meta Gemini 2.5 pro"""
    name = "gemini4"
    # previous gemini-2.5-pro-preview-05-06
    model = "gemini-2.5-pro-preview-06-05"
