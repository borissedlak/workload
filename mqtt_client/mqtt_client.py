import json
import time

import paho.mqtt.client as mqtt

import util


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        # Subscribe to the desired topic(s) here
        client.subscribe('tele/delock/SENSOR')
    else:
        print("Failed to connect, return code: ", rc)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    o = json.loads(payload)
    cons = o['ENERGY']['Power']

    print("Current Consumption: ", cons)
    util.write_to_blank_file(cons)
    util.get_consumption()


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection")

def setup():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    broker_address = "192.168.0.32"
    port = 1883  # Default MQTT port
    client.connect(broker_address, port=port)

    client.loop_start()

    # Run the network loop indefinitely
    while True:
        time.sleep(999999999)

    # Or run the network loop for a specific duration (e.g., 10 seconds)
    # import time
    # time.sleep(10)

    # Stop the network loop
    # client.loop_stop()