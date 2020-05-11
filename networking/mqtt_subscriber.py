import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Subscriber successfully connected")
    client.subscribe("po181u/#")

def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


subscriber = mqtt.Client("subscriber")

subscriber.username_pw_set("q0gpUtuYgoOiTTWeeVFmkcJiPDwN7OlmCpBmHV9N8NVRwFWzorLzsiF7L1JGDUS4")
subscriber.on_connect = on_connect
subscriber.on_message = on_message
subscriber.connect("mqtt.flespi.io", 1883, 60)

subscriber.loop_forever()