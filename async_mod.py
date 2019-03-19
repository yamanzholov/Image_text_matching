from logger_parser import info, warning, error
import random
import asyncio
from aiohttp import TCPConnector, ClientSession, ClientTimeout
from aiohttp import web_exceptions
import traceback


class AsyncParser:
    succ_c = 0
    fail_c = 0

    def __init__(self, proxy=None):
        self.proxy = proxy
        self.headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; rv:52.0) Gecko/20100101 Firefox/52.0",
            'Accept-Language': "en-US,en;q=0.5",
            'Content-Type': 'application/json; charset=utf-8'
        }

    async def fetch(self, url, session):
        """
        Async fetching.
        :param url:
        :param session: aiohttp client session
        :return:
        """
        # TODO: needs to be expanded in terms of exceptions. There needs to be more clarity of the errors with requests.
        try:
            if self.proxy:
                async with session.get(url, proxy=self.proxy, headers=self.headers) as response:
                    return await response.text(), url
            else:
                async with session.get(url, headers=self.headers) as response:
                    return await response.text(), url
        except web_exceptions.HTTPRequestTimeout:
            #error('Request to: ' + url + ', with proxy: ' + ' timed out.')
            return None, url
        except Exception as e:
            #error(str(traceback.format_exc()))
            #error(str(e))
            return None, url

    async def run(self, urls):
        """
        Async method which creates one instance of aiohttp.ClientSession and uses it to request multiple urls.
        :return:
        """
        # TODO: better logger info management. Needs statistics, sort of.
        tasks = []
        timeout = ClientTimeout(total=2 * 60, sock_connect=60, sock_read=30)
        connector = TCPConnector(verify_ssl=False, limit=1000)
        async with ClientSession(timeout=timeout, connector=connector) as s:
            for url in urls:
                task = asyncio.ensure_future(
                    self.fetch(url, s))
                tasks.append(task)
            responses = await asyncio.gather(*tasks)
        for response in responses:
            if response[0]:
                self.succ_c += 1
            else:
                # TODO: print out the status code. Or save the response as html and analyze. Something like that.
                self.fail_c += 1
        #info('Made ' + str(len(urls)) + ' requests')
        #info('Valid response count: ' + str(self.succ_c))
        #info('Invalid response count: ' + str(self.fail_c))
        self.fail_c = 0
        self.succ_c = 0
        return responses

    def start_async_parse(self, urls):
        """
        Initiates the crawling process of instagram users.
        :param chunk_size: cut the list into chunks with the appropriate size
        :return:
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(self.run(urls))
        results = loop.run_until_complete(future)
        return results
