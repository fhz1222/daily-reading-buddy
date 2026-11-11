import json
import threading
import unittest
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import server


class ReadBuddyTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(server.normalize(' Butterfly! '), 'butterfly')
        self.assertEqual(server.normalize('Good-morning.'), 'good morning')

    def test_local_health_and_static(self):
        web = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()
        try:
            host = f'http://127.0.0.1:{web.server_port}'
            with urlopen(host+'/') as res:
                self.assertIn('彩虹朗读小屋', res.read().decode())
            with urlopen(host+'/api/health') as res:
                data = json.load(res)
                self.assertEqual(data['mode'], 'local')
        finally:
            web.shutdown(); web.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
