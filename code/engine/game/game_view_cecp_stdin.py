import os
import sys
from threading import Thread

from engine.game import GameViewCECP


class GameViewCECPStdin(GameViewCECP):
    def __init__(self):
        super().__init__()
        self.__cmd_listener = Thread(target=self.__listen)
        self.__cmd_listener.start()

    def __listen(self):
        while not self.terminating:
            self._add_command(sys.stdin.readline().strip())

    def _send_command(self, cmd: str):
        sys.stdout.write(cmd + os.linesep)
