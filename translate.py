import sys
import re

import html.parser
import urllib.request
import urllib.parse
from async_mod import AsyncParser
import json
import time

class Translate:
    def __init__(self, ):
        self.parser = AsyncParser()
           
    def unescape(self, text):
        parser = html.parser.HTMLParser()
        return (parser.unescape(text))


    def translate(self, texts, to_language="auto", from_language="auto"):
        """Returns the translation using google translate
        you must shortcut the language you define
        (French = fr, English = en, Spanish = es, etc...)
        if not defined it will detect it or use english by default
        Example:
        print(translate("salut tu vas bien?", "en"))
        hello you alright?
        """
        t0 = time.time()
        base_link = "http://translate.google.com/m?hl=%s&sl=%s&q=%s"
        texts = [urllib.parse.quote(to_translate) for to_translate in texts]
        links = [base_link % (to_language, from_language, to_translate) for to_translate in texts]
        self.resps = self.parser.start_async_parse(links)
        data = [r[0] for r in self.resps]
        #resps = [json.loads(r[0]) for r in self.resps]
        #data = [d.decode("utf-8") for d in resps]
        expr = r'class="t0">(.*?)<'  
        re_result = [re.findall(expr, d) for d in data]
        results = []
        for result in re_result:
            if (len(result) == 0):
                result = ""
            else:
                result = self.unescape(result[0])
            results.append((result))
        t1 = time.time()
        print('TIME', t1-t0)
        return results