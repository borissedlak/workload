import http.client
import threading


class HttpClient:
    def __init__(self, HOST='localhost'):
        self.HOST = HOST
        self.PORT = 8080
        self.http_connection = None
        self.SYSTEM_STATS_PATH = "/system"
        self.APP_STATS_PATH = "/stats"
        self._open_connection()
        self.latest_config = 1
        self.ignore_response = False

    def _open_connection(self):
        print(f"Opening HTTP Connection with {self.HOST} and {self.PORT}")
        self.http_connection = http.client.HTTPConnection(self.HOST, self.PORT)

    def send_system_stats(self, cpu, device_name, disabled_aci, gpu_available):
        query_string = f"?cpu={cpu}&device_name={device_name}&disabled_aci={disabled_aci}&gpu_available={gpu_available}"
        self.http_connection.request("GET", self.SYSTEM_STATS_PATH + query_string)
        response = self.http_connection.getresponse()

    def send_app_stats(self, pixel, fps, pv, ra, threads, device_name):
        query_string = f"?pixel={pixel}&fps={fps}&pv={pv}&ra={ra}&threads={threads}&device_name={device_name}"
        self.http_connection.request("GET", self.APP_STATS_PATH + query_string)

        background_thread = threading.Thread(target=self._receive_in_other_thread())
        background_thread.daemon = True
        background_thread.start()

    def _receive_in_other_thread(self):
        response = self.http_connection.getresponse()
        # TODO: Check for status or throw exception
        response_text = response.read().decode('utf-8')
        c_threads, x = response_text.split(",")
        if not self.ignore_response:
            self.latest_config = int(c_threads)
        # print(f"Received {c_threads}")

    def get_latest_stream_config(self):
        return self.latest_config

    def override_stream_config(self, d_threads):
        self.ignore_response = True
        self.latest_config = d_threads
