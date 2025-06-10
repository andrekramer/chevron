"""Various models hosted by hugging face"""
import support

hugface_api_key = support.read_file_as_string("hugface-api-key").strip()

URL = "https://router.huggingface.co/hf-inference/models"
# was "https://api-inference.huggingface.co/models"

class HugFace(support.Model):
    """Gemma 2"""
    name = "hugface"
    model = "google/gemma-2-2b-it"
    text_field = "content"

    @classmethod
    def make_query(cls, text):
        """make a query for hugging face"""
        print("Hugging face model " + cls.model)
        return support.make_openai_std_query(text, cls.model)

    @classmethod
    async def ask(cls, session, query):
        """make a request to model hosted on hugging face"""
        url = URL + "/" + cls.model + "/v1/chat/completions"
        print("query url " + url)
        headers = {
          "Content-Type": "application/json",
          "Authorization": "Bearer " + hugface_api_key
        }
        return await support.ask(url, session, query, headers)

class HugFace2(HugFace):
    """Microsoft Phi mini 4k instruct"""
    name = "hugface2"
    model = "microsoft/Phi-3-mini-4k-instruct"

class HugFaceTooLarge(HugFace):
    """Microsoft Phi 4"""
    name = "hugface2"
    model = "microsoft/phi-4"

class HugFace3(HugFace2):
    """Qwen Qwen 2.5"""
    name = "hugface3"
    model = "Qwen/Qwen2.5-7B-Instruct"
