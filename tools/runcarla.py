# Import carla module
import carla

# Connect to the carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world object
world = client.get_world()

# Get the spectator object
spectator = world.get_spectator()
transform = ego_vehicle.get_transform()
spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                    carla.Rotation(pitch=-90)))


# Create a camera sensor blueprint
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# Set the camera attributes
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

# Spawn the camera sensor and attach it to the spectator
camera = world.spawn_actor(camera_bp, carla.Transform(), attach_to=spectator)

import pdb;pdb.set_trace()
# Start listening to the camera sensor
camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

# Do something with the spectator, such as moving or rotating it
# ...

# Stop listening to the camera sensor
camera.stop()

# Destroy the camera sensor
camera.destroy()