""" LECTURA import roslibpy
import time

client = roslibpy.Ros(host='localhost', port=9090)
client.run()

print("Connected:", client.is_connected)

topic = roslibpy.Topic(client, '/mobile_manipulator/joint_states' , 'sensor_msgs/JointState')

count = 0
start_time = None

def callback(msg):
    global count, start_time

    if start_time is None:
        start_time = time.time()

    count += 1
    elapsed = time.time() - start_time

    if elapsed > 0:
        hz = count / elapsed
        print(f"msgs: {count} | elapsed: {elapsed:.2f}s | hz: {hz:.2f}")

topic.subscribe(callback)

try:
    while client.is_connected:
        time.sleep(0.1)
except KeyboardInterrupt:
    topic.unsubscribe()
    client.terminate() """


""" COMANDO DE ESCRITURA DESDE DOCKER
rostopic pub  /mobile_manipulator/desired_traj mobile_manipulator_msgs/Trajectory "pose:
  translation:
    x: 1.0
    y: 0.0
    z: 0.5
joints: {mobjoint1: -10.0, mobjoint2: -10.0, mobjoint3: -10.0, joint1: -10.0, joint2: -10.0,
  joint3: -10.0, joint4: -10.0, joint5: -10.0, joint6: -10.0}" 
"""

#ESCRITURA
import roslibpy
import time

client = roslibpy.Ros(host='localhost', port=9090)
client.run()

print("Connected:", client.is_connected)

topic = roslibpy.Topic(
    client,
    '/mobile_manipulator/desired_traj',
    'mobile_manipulator_msgs/Trajectory'
)

topic.advertise()

msg = roslibpy.Message({
    "pose": {
        "translation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.5
        },
        "rotation": {
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "w": 0.5
        }
    },
    "vel": {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0}
    },
    "accel": {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0}
    },
    "joints": {
        "mobjoint1": -10.0,
        "mobjoint2": -10.0,
        "mobjoint3": -10.0,
        "joint1": -10.0,
        "joint2": -10.0,
        "joint3": -10.0,
        "joint4": -10.0,
        "joint5": -10.0,
        "joint6": -10.0
    }
})

topic.publish(msg)

time.sleep(1)

topic.unadvertise()
client.terminate()