import random as rand
import paho.mqtt.client as mqtt
import time

def temperature_simulator():
    offset = float(rand.randint(0, 80)) / 10
    return 18.0 + offset

def on_connect(client, userdata, flags, rc):
    print("Publisher successfully connected")

publisher=mqtt.Client("publisher")

publisher.username_pw_set("q0gpUtuYgoOiTTWeeVFmkcJiPDwN7OlmCpBmHV9N8NVRwFWzorLzsiF7L1JGDUS4")
publisher.on_connect = on_connect
publisher.connect("mqtt.flespi.io", 1883, 60)

publisher.loop_start()
while True:
    temperature = temperature_simulator()
    publisher.publish("po181u/Lab4", temperature)
    time.sleep(5)


