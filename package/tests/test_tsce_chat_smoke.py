import unittest
from tsce_agent_demo.tsce_chat import TSCEChat

class TestTSCEChatSmoke(unittest.TestCase):
    def test_tscechat_instantiation(self):
        chat = TSCEChat()
        self.assertIsNotNone(chat)

if __name__ == '__main__':
    unittest.main()
