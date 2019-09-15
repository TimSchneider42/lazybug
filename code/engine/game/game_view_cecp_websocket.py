from threading import Thread

import websocket

from engine.game import GameViewCECP


class GameViewCECPWebsocket(GameViewCECP):
    def __init__(self, connection_string: str):
        super().__init__()
        self.__run_client_thread = Thread(target=self.__run_client)
        self.__websocket = websocket.WebSocketApp(
            connection_string, on_message=self.__on_message, on_error=self.__on_error, on_close=self.__on_close)
        self.__websocket.on_open = self.__on_open
        self.__run_client_thread.start()

    def __run_client(self):
        self.__websocket.run_forever()

    def _send_command(self, cmd: str):
        print("Send: {}".format(cmd))
        self.__websocket.send(cmd)

    def terminate(self):
        super().terminate()
        self.__websocket.close()

    def __on_message(self, message):
        print("Recv: {}".format(message))
        self._add_command(message)

    def __on_error(self, error):
        print(error)
        self.terminate()

    def __on_close(self):
        print("### closed ###")

    def __on_open(self):
        print("Connection established.")
